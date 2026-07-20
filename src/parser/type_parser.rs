use std::collections::HashMap;
use std::mem;

use derive_ctor::ctor;
use itertools::Itertools;
use tree_sitter as ts;

use crate::utils::{TreeSitterUtils, matches_if};
use crate::{bytecode as b, context, errors};

#[derive(ctor)]
pub struct TypeParser<'a, 't> {
    #[ctor(default)]
    pub typedefs: Vec<DeclaredTypeDef<'t>>,
    #[ctor(default)]
    pub impls: Vec<(b::TypeRefKey, b::ImplDecl)>,
    #[ctor(expr(default_idents()))]
    pub idents: HashMap<String, b::TypeBody>,
    #[ctor(default)]
    pub typevar_count: usize,
    ctx: &'a context::BuildContext,
    src_idx: usize,
    mod_idx: usize,
}

impl<'a, 't> TypeParser<'a, 't> {
    pub fn finish(mut self, modules: &mut [b::Module]) -> Vec<b::TypeDef> {
        for (key, impl_decl) in mem::replace(&mut self.impls, vec![]) {
            self.finish_impl(key, impl_decl, modules);
        }
        self.typedefs.into_iter().map(|x| x.typedef).collect()
    }

    pub fn parse_type_expr(&self, node: ts::Node<'t>) -> b::Type {
        let node = node.of_kind("type_expr").child(0).unwrap();

        let body = match node.kind() {
            "ident" => self.parse_type_ident_with_args(node, []),
            "array_type" => {
                let item_ty = self.parse_type_expr(node.required_field("item_type"));
                b::TypeBody::builtin(b::BuiltinType::Array, [item_ty])
            }
            "generic_type" => {
                let name_node = node.required_field("name");
                let args = node
                    .iter_field("args")
                    .map(|arg_node| self.parse_type_expr(arg_node));
                self.parse_type_ident_with_args(name_node, args)
            }
            k => panic!("Unhandled type node `{k}`"),
        };
        b::Type::new(body, Some(b::Loc::from_node(self.src_idx, &node)))
    }

    pub fn parse_type_ident_with_args(
        &self,
        node: ts::Node<'t>,
        args: impl IntoIterator<Item = b::Type>,
    ) -> b::TypeBody {
        let mut args = args.into_iter().collect_vec();

        let ident =
            node.get_text(&self.ctx.source_manager.source(self.src_idx).content().text);

        macro_rules! validate_args {
            ($min:expr, $max:expr) => {{
                let (min, max) = ($min, $max);
                let expected_len = if args.len() < min {
                    Some(min)
                } else if args.len() > max {
                    Some(max)
                } else {
                    None
                };
                if let Some(expected_len) = expected_len {
                    self.ctx.push_error(errors::Error::new(
                        errors::WrongArgumentCount::new(
                            ident.to_string(),
                            expected_len,
                            args.len(),
                        )
                        .into(),
                        Some(b::Loc::from_node(self.src_idx, &node)),
                    ));
                    args.resize_with(expected_len, || b::Type::unknown(None));
                }
            }};
            ($count:expr) => {{
                let count = $count;
                validate_args!(count, count);
            }};
        }

        let mut body = self.parse_type_ident(node);

        match &mut body {
            b::TypeBody::TypeRef(type_ref) => {
                let modules = self.ctx.lock_modules();
                let decl = match type_ref.key {
                    b::TypeRefKey::Custom { mod_idx, idx } if mod_idx == self.mod_idx => {
                        &self.typedefs[idx].typedef
                    }
                    _ => type_ref.get_typedef(&*modules),
                };
                validate_args!(decl.generics.len() - type_ref.args.len());
                type_ref.args.extend(args);
            }
            body if body.is_unknown() => {}
            _ => validate_args!(0),
        }

        body
    }

    pub fn parse_type_ident(&self, node: ts::Node<'t>) -> b::TypeBody {
        let ident =
            node.get_text(&self.ctx.source_manager.source(self.src_idx).content().text);

        let Some(body) = self.idents.get(ident).cloned() else {
            self.ctx.push_error(errors::Error::new(
                errors::TypeNotFound::new(ident.to_string()).into(),
                Some(b::Loc::from_node(self.src_idx, &node)),
            ));
            return b::TypeBody::unknown();
        };

        body
    }

    pub fn parse_type_decl(&mut self, name: b::Name, node: ts::Node<'t>) {
        assert_eq!(node.kind(), "type_decl");

        let body_node = node.required_field("body");
        let body = match body_node.kind() {
            "record_type" => b::TypeDefBody::Record(b::RecordType::new()),
            "interface_type" => b::TypeDefBody::Interface,
            v => panic!("Unexpected type body kind: {v}"),
        };

        let value = b::TypeDef::new(
            name,
            body,
            b::Loc::from_node(self.src_idx, &node),
            node.iter_field("params").map(|_| b::TypeVarIdx::MAX),
        );
        self.idents.insert(
            value.name.last_ident().to_string(),
            b::TypeRef::new(b::TypeRefKey::Custom {
                mod_idx: self.mod_idx,
                idx:     self.typedefs.len(),
            })
            .into(),
        );
        self.typedefs.push(DeclaredTypeDef {
            typedef:        value,
            type_decl_node: Some(node),
        });
    }

    pub fn add_method(&mut self, ty: b::TypeRefKey, method: b::Method) -> usize {
        let modules = &mut self.ctx.lock_modules_mut();

        let typedef = match ty {
            b::TypeRefKey::Custom { mod_idx, idx } if mod_idx == self.mod_idx => {
                &mut self.typedefs[idx].typedef
            }
            b::TypeRefKey::Custom { mod_idx, idx } => {
                let module = &mut modules[mod_idx];
                &mut module.typedefs[idx]
            }
            b::TypeRefKey::Builtin(builtin_type) => {
                let module = &mut modules[b::BUILTINS_MODULE_IDX];
                module
                    .typedefs
                    .iter_mut()
                    .find(|td| {
                        matches_if!(
                            &td.body,
                            b::TypeDefBody::Builtin(b),
                            *b == builtin_type
                        )
                    })
                    .expect("builtin type not found")
            }
        };

        let new_method_idx = typedef.methods.len();
        typedef.methods.push(method);

        new_method_idx
    }

    pub fn get_typedef<'s>(
        &'s self,
        ty: b::TypeRefKey,
        modules: &'s [b::Module],
    ) -> &'s b::TypeDef {
        match ty {
            b::TypeRefKey::Custom { mod_idx, idx } if mod_idx == self.mod_idx => {
                &self.typedefs[idx].typedef
            }
            b::TypeRefKey::Builtin(needle) if self.mod_idx == b::BUILTINS_MODULE_IDX => {
                self.typedefs
                    .iter()
                    .map(|x| &x.typedef)
                    .find(|def| {
                        matches_if!(
                            &def.body,
                            &b::TypeDefBody::Builtin(builtin),
                            builtin == needle
                        )
                    })
                    .expect("builtin type not found")
            }

            _ => ty.get_typedef(modules),
        }
    }

    pub fn get_typedef_mut<'s>(
        &'s mut self,
        ty: b::TypeRefKey,
        modules: &'s mut [b::Module],
    ) -> &'s mut b::TypeDef {
        match ty {
            b::TypeRefKey::Custom { mod_idx, idx } if mod_idx == self.mod_idx => {
                &mut self.typedefs[idx].typedef
            }
            b::TypeRefKey::Builtin(needle) if self.mod_idx == b::BUILTINS_MODULE_IDX => {
                self.typedefs
                    .iter_mut()
                    .map(|x| &mut x.typedef)
                    .find(|def| {
                        matches_if!(
                            &def.body,
                            &b::TypeDefBody::Builtin(builtin),
                            builtin == needle
                        )
                    })
                    .expect("builtin type not found")
            }
            _ => ty.get_typedef_mut(modules),
        }
    }

    pub fn define_typedefs(&mut self) {
        for i in 0..self.typedefs.len() {
            self.define_typedef(i);
        }
    }

    fn define_typedef(&mut self, i: usize) {
        let ty_key = b::TypeRefKey::Custom {
            mod_idx: self.mod_idx,
            idx:     i,
        };

        let typedef = &self.typedefs[i];
        let Some(node) = typedef.type_decl_node else {
            return;
        };

        let generics = node
            .iter_field("params")
            .map(|param_node| {
                let ident = param_node.of_kind("ident").get_text(
                    &self.ctx.source_manager.source(self.src_idx).content().text,
                );

                if let Some(b::TypeBody::TypeVar(typevar)) = self.idents.get(ident)
                    && typevar.mod_idx == self.mod_idx
                {
                    return typevar.typevar_idx;
                }

                self.ctx.push_error(errors::Error::new(
                    errors::TypeVarNotFound::new(ident.to_string()).into(),
                    Some(b::Loc::from_node(self.src_idx, &param_node)),
                ));

                return b::TypeVarIdx::MAX;
            })
            .collect_vec();

        let body_node = node.required_field("body");
        let body = match (body_node.kind(), &typedef.typedef.body) {
            ("record_type", b::TypeDefBody::Record(rec)) => {
                let fields = body_node
                    .iter_field("fields")
                    .map(|field_node| {
                        let name_node = field_node.required_field("name");
                        let name = name_node
                            .get_text(
                                &self
                                    .ctx
                                    .source_manager
                                    .source(self.src_idx)
                                    .content()
                                    .text,
                            )
                            .to_string();
                        let record_field = b::RecordField::new(
                            name,
                            self.parse_type_expr(field_node.required_field("type")),
                            b::Loc::from_node(self.src_idx, &field_node),
                        );
                        record_field
                    })
                    .collect();

                b::TypeDefBody::Record(b::RecordType {
                    fields,
                    ..rec.clone()
                })
            }
            ("interface_type", b::TypeDefBody::Interface) => b::TypeDefBody::Interface,
            _ => unreachable!(),
        };

        let typedef = &mut self.typedefs[i];
        typedef.typedef.body = body;
        typedef.typedef.generics = generics;

        for ty_node in node.iter_field("implements") {
            let loc = b::Loc::from_node(self.src_idx, &ty_node);
            let ty = self.parse_type_expr(ty_node);
            let b::TypeBody::TypeRef(t) = ty.body else {
                self.ctx.push_error(errors::Error::new(
                    errors::TypeNotInterface::new(
                        &ty.body,
                        &self.ctx.lock_modules(),
                        &self.ctx.cfg,
                    )
                    .into(),
                    Some(b::Loc::from_node(self.src_idx, &node)),
                ));
                continue;
            };

            let iface_args = t.args.into_iter().map(|arg| arg.body).collect();
            let impl_decl = b::ImplDecl::new(t.key, iface_args, None, loc);
            self.impls.push((ty_key, impl_decl));
        }
    }

    fn finish_impl(
        &mut self,
        key: b::TypeRefKey,
        mut impl_decl: b::ImplDecl,
        modules: &mut [b::Module],
    ) {
        let method_names = self
            .get_typedef(impl_decl.iface, modules)
            .methods
            .iter()
            .map(|method| method.name.clone())
            .collect_vec();

        for method_name in method_names {
            self.finish_impl_method(key, &mut impl_decl, method_name, modules);
        }

        let typedef = self.get_typedef_mut(key, modules);
        typedef.impls.push(impl_decl)
    }

    fn finish_impl_method(
        &self,
        key: b::TypeRefKey,
        impl_decl: &mut b::ImplDecl,
        method_name: String,
        modules: &mut [b::Module],
    ) {
        let typedef = self.get_typedef(key, modules);

        let method_idx = typedef
            .methods
            .iter()
            .position(|method| method.name == method_name)
            .unwrap_or_else(|| {
                let iface_typedef = self.get_typedef(impl_decl.iface, modules);

                self.ctx.push_error(errors::Error::new(
                    errors::MethodNotImplemented::new(
                        method_name,
                        typedef
                            .name
                            .formated(modules, &self.ctx.cfg, Some(self.mod_idx)),
                        iface_typedef.name.formated(
                            modules,
                            &self.ctx.cfg,
                            Some(self.mod_idx),
                        ),
                    )
                    .into(),
                    Some(impl_decl.loc),
                ));

                usize::MAX // we have to add something
            });

        impl_decl.methods.push(method_idx);
    }
}

pub struct DeclaredTypeDef<'t> {
    pub typedef:        b::TypeDef,
    pub type_decl_node: Option<ts::Node<'t>>,
}

fn default_idents() -> HashMap<String, b::TypeBody> {
    let builtin = |builtin: b::BuiltinType| b::TypeBody::builtin(builtin, []);
    HashMap::from([
        ("void".to_string(), builtin(b::BuiltinType::Void)),
        ("never".to_string(), builtin(b::BuiltinType::Never)),
        ("bool".to_string(), builtin(b::BuiltinType::Bool)),
        ("i8".to_string(), builtin(b::BuiltinType::I8)),
        ("i16".to_string(), builtin(b::BuiltinType::I16)),
        ("i32".to_string(), builtin(b::BuiltinType::I32)),
        ("i64".to_string(), builtin(b::BuiltinType::I64)),
        ("u8".to_string(), builtin(b::BuiltinType::U8)),
        ("u16".to_string(), builtin(b::BuiltinType::U16)),
        ("u32".to_string(), builtin(b::BuiltinType::U32)),
        ("u64".to_string(), builtin(b::BuiltinType::U64)),
        ("usize".to_string(), builtin(b::BuiltinType::USize)),
        ("f32".to_string(), builtin(b::BuiltinType::F32)),
        ("f64".to_string(), builtin(b::BuiltinType::F64)),
        ("str".to_string(), builtin(b::BuiltinType::String)),
        ("Ptr".to_string(), builtin(b::BuiltinType::Ptr)),
    ])
}
