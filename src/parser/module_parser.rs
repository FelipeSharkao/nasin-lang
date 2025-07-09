use std::collections::HashMap;

use derive_new::new;
use itertools::{enumerate, multizip, Itertools};
use tree_sitter as ts;

use super::parser_value::ValueRef;
use super::type_parser::TypeParser;
use crate::parser::expr_parser::ExprParser;
use crate::parser::parser_value::ValueRefBody;
use crate::utils::TreeSitterUtils;
use crate::{bytecode as b, context, utils};

const UNDEF_VALUE: b::ValueIdx = usize::MAX;

const SELF_INDENT: &str = "Self";

#[derive(new)]
pub struct ModuleParser<'a, 't> {
    #[new(value = "TypeParser::new(ctx, src_idx, mod_idx)")]
    pub types:   TypeParser<'a, 't>,
    #[new(default)]
    pub globals: Vec<DeclaredGlobal<'t>>,
    #[new(default)]
    pub funcs:   Vec<DeclaredFunc<'t>>,
    #[new(default)]
    pub values:  Vec<b::Value>,
    #[new(default)]
    pub idents:  HashMap<String, ValueRef>,
    pub ctx:     &'a context::BuildContext,
    pub src_idx: usize,
    pub mod_idx: usize,
}

impl<'a, 't> ModuleParser<'a, 't> {
    pub fn finish(mut self) {
        for i in 0..self.globals.len() {
            let value_node = self.globals[i].value_node.clone();

            let mut value_parser = ExprParser::new(self, None, []);

            let value_ref = value_parser.add_expr_node(value_node, true);
            let result = value_parser.use_value_ref(&value_ref);

            (self, self.globals[i].global.body) = value_parser.finish(result);
            let global = &self.globals[i];
            if global.global.value == UNDEF_VALUE {
                let ty = global.ty.clone();
                let loc = global.global.loc;
                self.globals[i].global.value = self.create_value(ty, loc)
            }
        }

        for i in 0..self.funcs.len() {
            'parse: {
                let Some(value_node) = self.funcs[i].value_node else {
                    break 'parse;
                };
                let params = self.funcs[i].params.clone();

                let mut value_parser = ExprParser::new(self, Some(i), params);

                let value_ref = value_parser.add_expr_node(value_node, true);
                let result = value_parser.use_value_ref(&value_ref);

                (self, self.funcs[i].func.body) = value_parser.finish(result);
            };

            let func = &self.funcs[i];
            if func.func.ret == UNDEF_VALUE {
                let ret_ty = func.ret_ty.clone();
                let loc = func.func.loc;
                self.funcs[i].func.ret = self.create_value(ret_ty, loc)
            }
        }

        let typedefs = self.types.finish();

        let module = &mut self.ctx.lock_modules_mut()[self.mod_idx];
        module.typedefs = typedefs;
        module.globals = self.globals.into_iter().map(|x| x.global).collect();
        module.funcs = self.funcs.into_iter().map(|x| x.func).collect();
        module.values = self.values;
    }

    pub fn add_root(&mut self, node: ts::Node<'t>) {
        node.of_kind("root");

        for sym_node in node.iter_children() {
            let ident_node = sym_node.required_field("name").of_kind("ident");
            let name = ident_node
                .get_text(&self.ctx.source(self.src_idx).content().text)
                .to_string();

            match sym_node.kind() {
                "type_decl" => {
                    let mut old_idents = self.types.idents.clone();

                    let ty_idx = self.types.typedefs.len();
                    let self_type =
                        b::TypeRef::new(self.mod_idx, ty_idx).with_is_self(true);
                    self.types
                        .idents
                        .insert(SELF_INDENT.to_string(), self_type.into());

                    let generics = self.types.parse_generics(sym_node);
                    let generics_idents = generics.keys().copied().collect_vec();

                    let body_node = sym_node.required_field("body");
                    let is_virt = body_node.kind() == "interface_type";

                    let methods = body_node
                        .iter_field("methods")
                        .map(|method_node| {
                            let method_ident_node =
                                method_node.required_field("name").of_kind("ident");
                            let method_name = method_ident_node
                                .get_text(&self.ctx.source(self.src_idx).content().text);
                            let func_name = format!("{name}.{method_name}");

                            let func_idx = self.add_func(
                                func_name,
                                method_node,
                                Some((self.mod_idx, ty_idx, method_name.to_string())),
                                is_virt,
                            );

                            (method_name, (self.mod_idx, func_idx))
                        })
                        .collect();
                    self.types
                        .parse_type_decl(name, sym_node, methods, generics);

                    // TODO: move adding the methods to `finish_typedef` so we don't have
                    // to do this.
                    for name in generics_idents.into_iter().chain([SELF_INDENT]) {
                        if let Some(old_type) = old_idents.remove(name) {
                            self.types.idents.insert(name.to_string(), old_type);
                        } else {
                            self.types.idents.remove(name);
                        }
                    }
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
            if item.name.starts_with('_') {
                continue;
            }
            let ty_ref = b::TypeRef::new(mod_idx, i);
            self.types.idents.insert(item.name.clone(), ty_ref.into());
        }

        for (i, item) in enumerate(&module.funcs) {
            if item.name.starts_with('_') {
                continue;
            }
            let value = ValueRef::new(ValueRefBody::Func(mod_idx, i), item.loc);
            self.idents.insert(item.name.clone(), value);
        }

        for (i, item) in enumerate(&module.globals) {
            if item.name.starts_with('_') {
                continue;
            }
            let mut value = ValueRef::new(ValueRefBody::Global(mod_idx, i), item.loc);
            if item.body.len() == 1 {
                match &item.body[0].body {
                    b::InstrBody::CreateNumber(v) => {
                        value.body = ValueRefBody::Number(v.clone());
                    }
                    b::InstrBody::CreateBool(v) => {
                        value.body = ValueRefBody::Bool(*v);
                    }
                    _ => {}
                }
            }
            self.idents.insert(item.name.clone(), value);
        }
    }
    pub fn create_value(&mut self, ty: b::Type, loc: b::Loc) -> b::ValueIdx {
        self.values.push(b::Value::new(ty, loc));
        self.values.len() - 1
    }

    fn add_func(
        &mut self,
        name: String,
        node: ts::Node<'t>,
        method: Option<(usize, usize, String)>,
        is_virt: bool,
    ) -> usize {
        assert!(matches!(node.kind(), "func_decl" | "method"));

        let (params, params_names, params_locs): (Vec<_>, Vec<_>, Vec<_>) = node
            .iter_field("params")
            .map(|param_node| {
                let param_name_node = param_node.required_field("pat").of_kind("ident");
                let param_name = param_name_node
                    .get_text(&self.ctx.source(self.src_idx).content().text);

                let param_ty = match param_node.field("type") {
                    Some(ty_node) => self.types.parse_type_expr(ty_node),
                    None => b::Type::unknown(None),
                };

                let loc = b::Loc::from_node(self.src_idx, &param_node);
                (
                    self.create_value(param_ty, loc),
                    param_name.to_string(),
                    b::Loc::from_node(self.src_idx, &param_name_node),
                )
            })
            .multiunzip();

        let ret_ty = match node.field("ret_type") {
            Some(ty_node) => self.types.parse_type_expr(ty_node),
            None => b::Type::unknown(None),
        };

        let mut extrn: Option<b::Extern> = None;
        for directive_node in node.iter_field("directives") {
            let args_nodes: Vec<_> = directive_node.iter_field("args").collect();
            match directive_node
                .required_field("name")
                .get_text(&self.ctx.source(self.src_idx).content().text)
            {
                "extern" => {
                    // TODO: error handling
                    assert!(extrn.is_none());
                    assert!(args_nodes.len() == 1);
                    assert!(args_nodes[0].kind() == "string_lit");
                    let symbol_name = utils::decode_string_lit(
                        args_nodes[0]
                            .required_field("content")
                            .get_text(&self.ctx.source(self.src_idx).content().text),
                    );
                    extrn = Some(b::Extern { name: symbol_name });
                }
                _ => todo!(),
            }
        }

        let loc = b::Loc::from_node(self.src_idx, &node);
        let func = b::Func {
            name,
            params: params.clone(),
            ret: UNDEF_VALUE,
            method,
            extrn,
            is_virt,
            body: vec![],
            loc,
        };
        let func_idx = self.funcs.len();
        self.idents.insert(
            func.name.clone(),
            ValueRef::new(ValueRefBody::Func(self.mod_idx, func_idx), loc),
        );
        self.funcs.push(DeclaredFunc {
            func,
            value_node: node.field("return"),
            params: multizip((params_names, params, params_locs)).collect(),
            ret_ty,
        });

        func_idx
    }
    fn add_global(&mut self, name: String, node: ts::Node<'t>) {
        assert_eq!(node.kind(), "global_decl");

        let ty = match node.field("type") {
            Some(ty_node) => self.types.parse_type_expr(ty_node),
            None => b::Type::unknown(None),
        };

        let is_entry_point = name == "main";
        let global = b::Global {
            name,
            value: UNDEF_VALUE,
            body: vec![],
            is_entry_point,
            loc: b::Loc::from_node(self.src_idx, &node),
        };
        self.idents.insert(
            global.name.clone(),
            ValueRef::new(
                ValueRefBody::Global(self.mod_idx, self.globals.len()),
                b::Loc::from_node(self.src_idx, &node),
            ),
        );
        self.globals.push(DeclaredGlobal {
            global,
            value_node: node.required_field("value"),
            ty,
        });
    }
}

pub struct DeclaredFunc<'t> {
    pub func:   b::Func,
    value_node: Option<ts::Node<'t>>,
    params:     Vec<(String, b::ValueIdx, b::Loc)>,
    ret_ty:     b::Type,
}

pub struct DeclaredGlobal<'t> {
    pub global: b::Global,
    value_node: ts::Node<'t>,
    ty: b::Type,
}
