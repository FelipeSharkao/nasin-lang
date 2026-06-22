use std::collections::HashMap;

use derive_ctor::ctor;
use derive_more::Debug;
use itertools::{Itertools, enumerate};
use tree_sitter as ts;

use super::parser_value::ValueRef;
use super::type_parser::TypeParser;
use crate::parser::expr_parser::ExprParser;
use crate::parser::parser_value::ValueRefBody;
use crate::utils::TreeSitterUtils;
use crate::{bytecode as b, context, errors, utils};

const UNDEF_VALUE: b::ValueIdx = usize::MAX;

const SELF_TYPE_INDENT: &str = "Self";

#[derive(ctor)]
pub struct ModuleParser<'a, 't> {
    #[ctor(expr(TypeParser::new(ctx, src_idx, mod_idx)))]
    pub types:        TypeParser<'a, 't>,
    #[ctor(default)]
    pub globals:      Vec<DeclaredGlobal<'t>>,
    #[ctor(default)]
    pub funcs:        Vec<DeclaredFunc<'t>>,
    #[ctor(default)]
    pub values:       Vec<b::Value>,
    #[ctor(default)]
    pub blocks:       Vec<b::Block>,
    #[ctor(default)]
    pub idents:       HashMap<String, ValueRef>,
    #[ctor(default)]
    pub typevar_defs: Vec<b::TypeVarDef>,
    pub ctx:          &'a context::BuildContext,
    pub src_idx:      usize,
    pub mod_idx:      usize,
}

impl<'a, 't> ModuleParser<'a, 't> {
    pub fn finish(mut self) {
        self.types.define_typedefs();

        for i in 0..self.globals.len() {
            let value_node = self.globals[i].value_node;
            let block_idx = self.globals[i].global.body;

            let mut value_parser = ExprParser::new(self, None, block_idx);
            value_parser.add_expr_node(value_node, Some(block_idx));
            self = value_parser.finish();

            let global = &self.globals[i];
            if global.global.value == UNDEF_VALUE {
                let ty = global.ty.clone();
                let loc = global.global.loc;
                self.globals[i].global.value = self.create_value(ty, Some(loc))
            }
        }

        for i in 0..self.funcs.len() {
            self.define_func(i);
        }

        let typedefs = self.types.finish();

        let module = &mut self.ctx.lock_modules_mut()[self.mod_idx];
        module.typedefs = typedefs;
        module.typevars = self.typevar_defs;
        module.globals = self.globals.into_iter().map(|x| x.global).collect();
        module.funcs = self.funcs.into_iter().map(|x| x.func).collect();
        module.blocks = self.blocks;
        module.values = self.values;
    }

    pub fn add_root(&mut self, node: ts::Node<'t>) {
        node.of_kind("root");

        for decl_node in node.iter_children() {
            let get_name = || {
                let name_node = decl_node.required_field("name").of_kind("ident");

                let name_kind = match decl_node.kind() {
                    "global_decl" => b::NameIdentKind::Value,
                    "func_decl" => b::NameIdentKind::Func,
                    "type_decl" | "typevar_decl" | "impl_decl" => b::NameIdentKind::Type,
                    _ => panic!("Unexpected declaration kind: {}", decl_node.kind()),
                };

                self.ctx.lock_modules()[self.mod_idx].name.with(
                    name_node.get_text(
                        &self.ctx.source_manager.source(self.src_idx).content().text,
                    ),
                    name_kind,
                    Some(b::Loc::from_node(self.src_idx, &name_node)),
                )
            };

            match decl_node.kind() {
                "type_decl" => self.declare_type(get_name(), decl_node),
                "typevar_decl" => self.declare_typevar(get_name(), decl_node),
                "func_decl" => self.declare_func(get_name(), decl_node, None, false),
                "global_decl" => self.declare_global(get_name(), decl_node),
                "impl_decl" => self.declare_impl(decl_node),
                "comment" => {}
                _ => panic!("Unexpected declaration kind: {}", decl_node.kind()),
            }
        }
    }

    pub fn open_module(&mut self, mod_idx: usize) {
        let module = &self.ctx.lock_modules()[mod_idx];

        for (i, item) in enumerate(&module.typedefs) {
            let ty_ref = b::TypeRef::new(b::TypeRefKey::Custom { mod_idx, idx: i });
            self.types
                .idents
                .insert(item.name.last_ident().to_string(), ty_ref.into());
        }

        for (i, item) in enumerate(&module.funcs) {
            let value = ValueRef::new(ValueRefBody::Func(mod_idx, i), item.loc);
            self.idents
                .insert(item.name.last_ident().to_string(), value);
        }

        for (i, item) in enumerate(&module.globals) {
            let mut value =
                ValueRef::new(ValueRefBody::Global(mod_idx, i), Some(item.loc));
            let body = &module.blocks[item.body].body;
            if body.len() == 1 {
                match &body[0].body {
                    b::InstrBody::CreateNumber(v) => {
                        value.body = ValueRefBody::Number(v.clone());
                    }
                    b::InstrBody::CreateBool(v) => {
                        value.body = ValueRefBody::Bool(*v);
                    }
                    _ => {}
                }
            }
            self.idents
                .insert(item.name.last_ident().to_string(), value);
        }
    }

    pub fn create_value(&mut self, ty: b::Type, loc: Option<b::Loc>) -> b::ValueIdx {
        self.values.push(b::Value::new(ty, loc));
        self.values.len() - 1
    }

    pub fn add_block(&mut self) -> b::BlockIdx {
        self.blocks.push(b::Block::default());
        self.blocks.len() - 1
    }

    fn declare_func(
        &mut self,
        name: b::Name,
        node: ts::Node<'t>,
        method_info: Option<b::FuncMethodInfo>,
        is_virt: bool,
    ) {
        assert!(matches!(node.kind(), "func_decl" | "func_sig"));

        let loc = b::Loc::from_node(self.src_idx, &node);

        let (name, method_info) = if let Some(parent) = node.field("parent") {
            assert!(parent.kind() == "type_expr");

            let parent_ty = self.types.parse_type_expr(parent);
            let b::TypeBody::TypeRef(ty_ref) = &parent_ty.body else {
                self.ctx.push_error(errors::Error::new(
                    errors::Todo::new("method for internal type".to_string()).into(),
                    Some(b::Loc::from_node(self.src_idx, &parent)),
                ));
                return;
            };

            let method_name = name.last_ident().to_string();

            let mut method_info = b::FuncMethodInfo::new(
                method_name.clone(),
                ty_ref.key,
                // FIXME: not all methods are virtual, we should handle this properly
                true,
            );
            method_info.ty_args = ty_ref.args.clone();

            let modules = self.ctx.lock_modules();
            let typedef = self.types.get_typedef(ty_ref.key, &*modules);

            (
                typedef
                    .name
                    .with(method_name, b::NameIdentKind::Func, Some(loc)),
                Some(method_info),
            )
        } else {
            (name, method_info)
        };

        let params = node
            .iter_field("params")
            .map(|param_node| {
                let param_name_node = param_node.required_field("pat").of_kind("ident");
                let param_name = param_name_node.get_text(
                    &self.ctx.source_manager.source(self.src_idx).content().text,
                );

                let loc = b::Loc::from_node(self.src_idx, &param_node);
                DeclaredParam::new(
                    param_name.to_string(),
                    self.create_value(b::Type::unknown(None), Some(loc)),
                    b::Loc::from_node(self.src_idx, &param_name_node),
                    param_node.field("type"),
                )
            })
            .collect_vec();

        let ret = self.create_value(b::Type::unknown(None), Some(loc));

        let mut extrn = None;
        let mut is_entry = false;
        for directive_node in node.iter_field("directives") {
            let args_nodes: Vec<_> = directive_node.iter_field("args").collect();

            let ident = directive_node
                .required_field("name")
                .get_text(&self.ctx.source_manager.source(self.src_idx).content().text);

            macro_rules! validate_args {
                ($count:expr) => {{
                    let count = $count;
                    if args_nodes.len() != count {
                        self.ctx.push_error(errors::Error::new(
                            errors::WrongArgumentCount::new(
                                ident.to_string(),
                                count,
                                args_nodes.len(),
                            )
                            .into(),
                            Some(b::Loc::from_node(self.src_idx, &directive_node)),
                        ));
                        continue;
                    }
                }};
            }

            match ident {
                "extern" => {
                    // TODO: error handling
                    validate_args!(1);
                    assert!(args_nodes[0].kind() == "string_lit");
                    let symbol_name = utils::decode_string_lit(
                        args_nodes[0].required_field("content").get_text(
                            &self.ctx.source_manager.source(self.src_idx).content().text,
                        ),
                    )
                    .to_string();
                    extrn = Some(b::Extern { name: symbol_name });
                }
                "entry" => {
                    validate_args!(0);
                    is_entry = true;
                }
                _ => todo!(),
            }
        }

        let func = b::Func {
            name,
            params: params.iter().map(|x| x.value).collect(),
            ret,
            method: method_info.clone(),
            extrn,
            is_entry,
            is_virt,
            body: self.add_block(),
            loc: Some(loc),
            generics: vec![],
            generic_instantiations: HashMap::new(),
        };

        let func_idx = self.funcs.len();
        self.idents.insert(
            func.name.last_ident().to_string(),
            ValueRef::new(ValueRefBody::Func(self.mod_idx, func_idx), Some(loc)),
        );

        self.funcs.push(DeclaredFunc::new(
            func,
            params,
            node.field("return"),
            node.field("ret_type"),
        ));

        if let Some(method_info) = method_info {
            let method = b::Method::new((self.mod_idx, func_idx), loc);
            self.types
                .add_method(method_info.ty, method_info.name, method);
        }
    }

    fn declare_global(&mut self, name: b::Name, node: ts::Node<'t>) {
        assert_eq!(node.kind(), "global_decl");

        let ty = match node.field("type") {
            Some(ty_node) => self.types.parse_type_expr(ty_node),
            None => b::Type::unknown(None),
        };

        let is_main = name.last_ident() == "main";

        let global = b::Global {
            name,
            value: UNDEF_VALUE,
            body: self.add_block(),
            loc: b::Loc::from_node(self.src_idx, &node),
        };
        self.idents.insert(
            global.name.last_ident().to_string(),
            ValueRef::new(
                ValueRefBody::Global(self.mod_idx, self.globals.len()),
                Some(b::Loc::from_node(self.src_idx, &node)),
            ),
        );
        self.globals.push(DeclaredGlobal {
            global,
            value_node: node.required_field("value"),
            ty,
        });

        if is_main {
            let mut main = self.ctx.main.write().unwrap();
            *main = Some((self.mod_idx, self.globals.len() - 1));
        }
    }

    fn declare_type(&mut self, name: b::Name, node: ts::Node<'t>) {
        let ty_idx = self.types.typedefs.len();

        let body_node = node.required_field("body");
        let is_virt = body_node.kind() == "interface_type";

        self.types.parse_type_decl(name.clone(), node);

        for method_node in body_node.iter_field("methods") {
            let method_name_node = method_node.required_field("name").of_kind("ident");
            let method_name = method_name_node
                .get_text(&self.ctx.source_manager.source(self.src_idx).content().text);

            self.declare_func(
                name.with(
                    method_name,
                    b::NameIdentKind::Func,
                    Some(b::Loc::from_node(self.src_idx, &method_name_node)),
                ),
                method_node,
                Some(b::FuncMethodInfo::new(
                    method_name.to_string(),
                    b::TypeRefKey::Custom {
                        mod_idx: self.mod_idx,
                        idx:     ty_idx,
                    },
                    // FIXME: since in the method's implementations we're not
                    // handling is_virtual properly, we can't handle it here
                    // as well to be consistent. As soon as we implement it
                    // there, we should use `is_virt` here
                    true,
                )),
                is_virt,
            );
        }
    }

    fn declare_typevar(&mut self, name: b::Name, node: ts::Node<'t>) {
        let typevar_idx = self.types.typevar_count;
        self.types.typevar_count += 1;
        let typevar_def = b::TypeVarDef::new(
            name.clone(),
            node.field("constraint")
                .map(|ty_node| self.types.parse_type_expr(ty_node)),
            Some(b::Loc::from_node(self.src_idx, &node)),
        );
        self.typevar_defs.push(typevar_def);
        self.types.idents.insert(
            name.last_ident().to_string(),
            b::TypeVar::new(self.mod_idx, typevar_idx).into(),
        );
    }

    fn declare_impl(&mut self, node: ts::Node<'t>) {
        let loc = b::Loc::from_node(self.src_idx, &node);

        let ty_node = node.required_field("type").of_kind("type_expr");

        let ty = self.types.parse_type_expr(ty_node);
        if ty.is_unknown() {
            // parse_type_expr already pushed an error
            return;
        }

        let b::TypeBody::TypeRef(ty_ref) = &ty.body else {
            self.ctx.push_error(errors::Error::new(
                errors::Todo::new("impl for internal type".to_string()).into(),
                Some(b::Loc::from_node(self.src_idx, &ty_node)),
            ));
            return;
        };

        let constraints = if ty_ref.args.is_empty() {
            None
        } else {
            Some(ty_ref.args.iter().map(|arg| arg.body.clone()).collect())
        };

        let modules = &mut self.ctx.lock_modules_mut();

        let iface_keys = node
            .iter_field("implements")
            .filter_map(|iface_node| {
                let iface_ty = self.types.parse_type_ident(iface_node);
                if iface_ty.is_unknown() {
                    // parse_type_ident already pushed an error
                    return None;
                }

                let b::TypeBody::TypeRef(iface_ty_ref) = iface_ty else {
                    self.ctx.push_error(errors::Error::new(
                        errors::TypeNotInterface::new(&iface_ty, modules, &self.ctx.cfg)
                            .into(),
                        Some(b::Loc::from_node(self.src_idx, &iface_node)),
                    ));
                    return None;
                };

                Some(iface_ty_ref.key)
            })
            .collect_vec();

        let typedef = self.types.get_typedef_mut(ty_ref.key, modules);
        for iface_key in iface_keys {
            let impl_decl = b::ImplDecl::new(iface_key, constraints.clone(), loc);
            typedef.impls.push(impl_decl);
        }
    }

    fn define_func(&mut self, i: usize) {
        let func = &mut self.funcs[i];

        let old_self_type = self.types.idents.get(SELF_TYPE_INDENT).cloned();

        if let Some(method) = &func.func.method {
            let type_ref = b::TypeRef::new(method.ty).with_args(method.ty_args.clone());
            self.types
                .idents
                .insert(SELF_TYPE_INDENT.to_string(), type_ref.into());
        }

        for param in &func.params {
            if let Some(ty_node) = param.ty_node {
                self.values[param.value].ty = self.types.parse_type_expr(ty_node);
            }
        }

        if let Some(ret_ty_node) = func.ret_ty_node {
            let ty = self.types.parse_type_expr(ret_ty_node);
            self.values[func.func.ret].ty = ty;
        }

        func.func.generics = func
            .params
            .iter()
            .flat_map(|param| {
                let param_ty = &self.values[param.value].ty;
                param_ty.typevars()
            })
            .chain(self.values[func.func.ret].ty.typevars())
            .unique()
            .sorted()
            .collect();

        if let Some(value_node) = func.value_node {
            let block_idx = func.func.body;

            utils::replace_with(self, |module| {
                let mut value_parser = ExprParser::new(module, Some(i), block_idx);
                value_parser.add_expr_node(value_node, Some(block_idx));
                value_parser.finish()
            });
        }

        if let Some(old_self_type) = old_self_type {
            self.types
                .idents
                .insert(SELF_TYPE_INDENT.to_string(), old_self_type);
        } else {
            self.types.idents.remove(SELF_TYPE_INDENT);
        }
    }
}

#[derive(Debug, ctor)]
pub struct DeclaredFunc<'t> {
    pub func:    b::Func,
    pub params:  Vec<DeclaredParam<'t>>,
    value_node:  Option<ts::Node<'t>>,
    ret_ty_node: Option<ts::Node<'t>>,
}

#[derive(Debug, ctor)]
pub struct DeclaredParam<'t> {
    pub name:  String,
    pub value: b::ValueIdx,
    pub loc:   b::Loc,
    #[debug(skip)]
    ty_node:   Option<ts::Node<'t>>,
}

#[derive(Debug, ctor)]
pub struct DeclaredGlobal<'t> {
    pub global: b::Global,
    value_node: ts::Node<'t>,
    ty:         b::Type,
}
