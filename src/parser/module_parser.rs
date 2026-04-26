use std::collections::HashMap;

use derive_ctor::ctor;
use derive_more::Debug;
use itertools::{Itertools, enumerate};
use tree_sitter as ts;

use super::parser_value::ValueRef;
use super::type_parser::TypeParser;
use crate::parser::expr_parser::ExprParser;
use crate::parser::parser_value::ValueRefBody;
use crate::parser::type_parser::UNDEF_TYPEVAR;
use crate::utils::TreeSitterUtils;
use crate::{bytecode as b, context, errors, utils};

const UNDEF_VALUE: b::ValueIdx = usize::MAX;

const SELF_TYPE_INDENT: &str = "Self";

#[derive(ctor)]
pub struct ModuleParser<'a, 't> {
    #[ctor(expr(TypeParser::new(ctx, src_idx, mod_idx)))]
    pub types: TypeParser<'a, 't>,
    #[ctor(default)]
    pub globals: Vec<DeclaredGlobal<'t>>,
    #[ctor(default)]
    pub funcs: Vec<DeclaredFunc<'t>>,
    #[ctor(default)]
    pub values: Vec<b::Value>,
    #[ctor(default)]
    pub blocks: Vec<b::Block>,
    #[ctor(default)]
    pub idents: HashMap<String, ValueRef>,
    #[ctor(default)]
    pub typevar_defs: Vec<b::TypeVarDef>,
    pub ctx: &'a context::BuildContext,
    pub src_idx: usize,
    pub mod_idx: usize,
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

        for sym_node in node.iter_children() {
            let name_node = sym_node.required_field("name").of_kind("ident");

            let name_kind = match sym_node.kind() {
                "global_decl" => b::NameIdentKind::Value,
                "func_decl" => b::NameIdentKind::Func,
                "type_decl" | "typevar_decl" => b::NameIdentKind::Type,
                _ => panic!("Unexpected symbol kind: {}", sym_node.kind()),
            };

            let name = {
                self.ctx.lock_modules()[self.mod_idx].name.with(
                    name_node.get_text(
                        &self.ctx.source_manager.source(self.src_idx).content().text,
                    ),
                    name_kind,
                    Some(b::Loc::from_node(self.src_idx, &name_node)),
                )
            };

            match sym_node.kind() {
                "type_decl" => {
                    let ty_idx = self.types.typedefs.len();

                    let body_node = sym_node.required_field("body");
                    let is_virt = body_node.kind() == "interface_type";

                    self.types.parse_type_decl(name.clone(), sym_node);

                    for method_node in body_node.iter_field("methods") {
                        let method_name_node =
                            method_node.required_field("name").of_kind("ident");
                        let method_name = method_name_node.get_text(
                            &self.ctx.source_manager.source(self.src_idx).content().text,
                        );

                        self.add_func(
                            name.with(
                                method_name,
                                b::NameIdentKind::Func,
                                Some(b::Loc::from_node(self.src_idx, &method_name_node)),
                            ),
                            method_node,
                            Some(b::FuncMethodInfo::new(
                                method_name.to_string(),
                                self.mod_idx,
                                ty_idx,
                            )),
                            is_virt,
                        );
                    }
                }
                "typevar_decl" => {
                    let typevar_idx = self.types.typevar_count;
                    self.types.typevar_count += 1;
                    let typevar_def = b::TypeVarDef {
                        name: name.clone(),
                        loc:  b::Loc::from_node(self.src_idx, &sym_node),
                    };
                    self.typevar_defs.push(typevar_def);
                    self.types.idents.insert(
                        name.last_ident().to_string(),
                        b::TypeVar::new(self.mod_idx, typevar_idx).into(),
                    );
                }
                "func_decl" => {
                    self.add_func(name, sym_node, None, false);
                }
                "global_decl" => {
                    self.add_global(name, sym_node);
                }
                _ => panic!("Unexpected symbol kind: {}", sym_node.kind()),
            }
        }
    }

    pub fn open_module(&mut self, mod_idx: usize) {
        let module = &self.ctx.lock_modules()[mod_idx];

        for (i, item) in enumerate(&module.typedefs) {
            let ty_ref = b::TypeRef::new(mod_idx, i);
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

    fn add_func(
        &mut self,
        name: b::Name,
        node: ts::Node<'t>,
        method_info: Option<b::FuncMethodInfo>,
        is_virt: bool,
    ) {
        assert!(matches!(node.kind(), "func_decl" | "func_sig"));

        let loc = b::Loc::from_node(self.src_idx, &node);

        let (name, method_info) = if let Some(parent) = node.field("parent") {
            assert!(parent.kind() == "ident");

            let parent_ty = self.types.parse_type_ident(parent);
            let b::TypeBody::TypeRef(ty_ref) = parent_ty else {
                self.ctx.push_error(errors::Error::new(
                    errors::Todo::new("method for builtin type".to_string()).into(),
                    Some(b::Loc::from_node(self.src_idx, &parent)),
                ));
                return;
            };

            let method_name = name.last_ident().to_string();

            let method_info =
                b::FuncMethodInfo::new(method_name.clone(), ty_ref.mod_idx, ty_ref.idx);

            let modules = self.ctx.lock_modules();
            (
                self.types
                    .get_type_name(ty_ref.mod_idx, ty_ref.idx, &*modules)
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

        let mut extrn: Option<b::Extern> = None;
        for directive_node in node.iter_field("directives") {
            let args_nodes: Vec<_> = directive_node.iter_field("args").collect();
            match directive_node
                .required_field("name")
                .get_text(&self.ctx.source_manager.source(self.src_idx).content().text)
            {
                "extern" => {
                    // TODO: error handling
                    assert!(extrn.is_none());
                    assert!(args_nodes.len() == 1);
                    assert!(args_nodes[0].kind() == "string_lit");
                    let symbol_name = utils::decode_string_lit(
                        args_nodes[0].required_field("content").get_text(
                            &self.ctx.source_manager.source(self.src_idx).content().text,
                        ),
                    );
                    extrn = Some(b::Extern { name: symbol_name });
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
            is_entry: false,
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
            self.types.add_method(
                method_info.mod_idx,
                method_info.ty_idx,
                method_info.name,
                b::Method::new((self.mod_idx, func_idx), loc),
            );
        }
    }

    fn add_global(&mut self, name: b::Name, node: ts::Node<'t>) {
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

    fn define_func(&mut self, i: usize) {
        let func = &mut self.funcs[i];

        let old_self_type = self.types.idents.get(SELF_TYPE_INDENT).cloned();

        let self_ty_ref = if let Some(method) = &func.func.method {
            let type_def = &self.types.typedefs[method.ty_idx].typedef;

            let args = type_def.generics.iter().map(|&idx| {
                assert!(idx < UNDEF_TYPEVAR);
                b::Type::new(b::TypeVar::new(self.mod_idx, idx).into(), None)
            });
            let type_ref = b::TypeRef::new(method.mod_idx, method.ty_idx)
                .with_args(args.collect_vec());

            self.types
                .idents
                .insert(SELF_TYPE_INDENT.to_string(), type_ref.clone().into());
            Some(type_ref)
        } else {
            None
        };

        for (i, param) in func.params.iter().enumerate() {
            if let Some(ty_node) = param.ty_node {
                let mut ty = self.types.parse_type_expr(ty_node);
                if let b::TypeBody::TypeRef(ty_ref) = &mut ty.body
                    && self_ty_ref.as_ref().is_some_and(|x| ty_ref.is_same_of(x))
                    && i == 0
                {
                    ty_ref.is_self = true;
                }

                self.values[param.value].ty = ty;
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
    ty: b::Type,
}
