use std::collections::{HashMap, HashSet};

use derive_ctor::ctor;
use itertools::Itertools;
use tree_sitter as ts;

use crate::utils::{IntoItem, SortedMap, TreeSitterUtils};
use crate::{bytecode as b, context, errors};

pub const UNDEF_TYPEVAR: b::TypeVarIdx = usize::MAX;

#[derive(ctor)]
pub struct TypeParser<'a, 't> {
    #[ctor(default)]
    pub typedefs: Vec<DeclaredTypeDef<'a, 't>>,
    #[ctor(expr(default_idents()))]
    pub idents: HashMap<String, b::TypeBody>,
    #[ctor(default)]
    pub typevar_count: usize,
    ctx: &'a context::BuildContext,
    src_idx: usize,
    mod_idx: usize,
}

impl<'a, 't> TypeParser<'a, 't> {
    pub fn finish(self) -> Vec<b::TypeDef> {
        self.typedefs.into_iter().map(|x| x.typedef).collect()
    }

    pub fn parse_type_expr(&self, node: ts::Node<'t>) -> b::Type {
        let node = node.of_kind("type_expr").child(0).unwrap();

        let body = match node.kind() {
            "ident" => self.parse_type_ident(node, []),
            "array_type" => {
                let item_ty = self.parse_type_expr(node.required_field("item_type"));
                b::TypeBody::Array(item_ty.into())
            }
            "generic_type" => {
                let name_node = node.required_field("name");
                let args = node
                    .iter_field("args")
                    .map(|arg_node| self.parse_type_expr(arg_node));
                self.parse_type_ident(name_node, args)
            }
            k => panic!("Unhandled type node `{k}`"),
        };
        b::Type::new(body, Some(b::Loc::from_node(self.src_idx, &node)))
    }

    pub fn parse_type_ident(
        &self,
        node: ts::Node<'t>,
        args: impl IntoIterator<Item = b::Type>,
    ) -> b::TypeBody {
        let mut args = args.into_iter().collect_vec();

        let ident =
            node.get_text(&self.ctx.source_manager.source(self.src_idx).content().text);

        macro_rules! validate_args {
            ($min:expr, $max:expr) => {{
                let min = $min;
                let max = $max;
                let len = args.len();
                let expected_len = if len < min {
                    Some(min)
                } else if len > max {
                    Some(max)
                } else {
                    None
                };
                if let Some(expected_len) = expected_len {
                    self.ctx.push_error(errors::Error::new(
                        errors::WrongArgumentCount::new(
                            ident.to_string(),
                            expected_len,
                            len,
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

        let Some(mut body) = self.idents.get(ident).cloned() else {
            self.ctx.push_error(errors::Error::new(
                errors::TypeNotFound::new(ident.to_string()).into(),
                Some(b::Loc::from_node(self.src_idx, &node)),
            ));
            return b::TypeBody::unknown();
        };

        match &mut body {
            b::TypeBody::TypeRef(type_ref) if !type_ref.is_self => {
                let modules = self.ctx.lock_modules();
                let decl = if type_ref.mod_idx == self.mod_idx {
                    &self.typedefs[type_ref.idx].typedef
                } else {
                    &modules[type_ref.mod_idx].typedefs[type_ref.idx]
                };
                validate_args!(decl.generics.len());
                type_ref.args = args;
            }
            b::TypeBody::Ptr(item_ty @ None) => {
                validate_args!(0, 1);
                *item_ty = args.into_item(0).map(|ty| ty.into());
            }
            body if body.is_unknown() => {}
            _ => validate_args!(0),
        }

        body
    }

    pub fn parse_type_decl(
        &mut self,
        name: b::Name,
        node: ts::Node<'t>,
        methods_idx: HashMap<&'a str, (usize, usize)>,
    ) {
        assert_eq!(node.kind(), "type_decl");

        let body_node = node.required_field("body");
        let body = match body_node.kind() {
            "record_type" => b::TypeDefBody::Record(b::RecordType {
                fields:  SortedMap::new(),
                methods: SortedMap::new(),
                ifaces:  HashSet::new(),
            }),
            "interface_type" => b::TypeDefBody::Interface(b::InterfaceType {
                methods: SortedMap::new(),
            }),
            v => panic!("Unexpected type body kind: {v}"),
        };

        let generics = node
            .iter_field("params")
            .map(|_| b::TypeVarIdx::MAX)
            .collect_vec();

        let value = b::TypeDef {
            name,
            body,
            loc: b::Loc::from_node(self.src_idx, &node),
            generics,
        };
        self.idents.insert(
            value.name.last_ident().to_string(),
            b::TypeRef::new(self.mod_idx, self.typedefs.len()).into(),
        );
        self.typedefs.push(DeclaredTypeDef {
            typedef: value,
            type_decl_node: Some(node),
            methods_idx,
        });
    }

    pub fn define_typedefs(&mut self) {
        for i in 0..self.typedefs.len() {
            self.define_typedef(i);
        }
    }

    fn define_typedef(&mut self, i: usize) {
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
        let body =
            match (body_node.kind(), &typedef.typedef.body) {
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
                                name.clone(),
                                self.parse_type_expr(field_node.required_field("type")),
                                b::Loc::from_node(self.src_idx, &field_node),
                            );
                            (name, record_field)
                        })
                        .collect();

                    let methods = body_node
                        .iter_field("methods")
                        .map(|method_node| {
                            let name_node = method_node.required_field("name");
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
                            let func_ref = typedef.methods_idx.get(&name as &str).expect(
                                "index of method's function should already be known",
                            );

                            let method = b::Method::new(
                                name.clone(),
                                *func_ref,
                                b::Loc::from_node(self.src_idx, &method_node),
                            );
                            (name, method)
                        })
                        .collect();

                    let implements = node
                        .iter_field("assertion")
                        .map(|ty_node| self.parse_type_expr(ty_node))
                        .filter_map(|ty| match ty.body {
                            b::TypeBody::TypeRef(t) => Some((t.mod_idx, t.idx)),
                            _ => {
                                self.ctx.push_error(errors::Error::new(
                                    errors::TypeNotInterface::new(
                                        &ty,
                                        &self.ctx.lock_modules(),
                                        &self.ctx.cfg,
                                    )
                                    .into(),
                                    Some(b::Loc::from_node(self.src_idx, &node)),
                                ));
                                None
                            }
                        })
                        .collect();

                    b::TypeDefBody::Record(b::RecordType {
                        fields,
                        methods,
                        ifaces: implements,
                        ..rec.clone()
                    })
                }
                ("interface_type", b::TypeDefBody::Interface(iface)) => {
                    let methods = body_node
                        .iter_field("methods")
                        .map(|method_node| {
                            let name_node = method_node.required_field("name");
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
                            let func_ref = typedef.methods_idx.get(&name as &str).expect(
                                "index of method's function should already be known",
                            );

                            let method = b::Method::new(
                                name.clone(),
                                *func_ref,
                                b::Loc::from_node(self.src_idx, &method_node),
                            );
                            (name, method)
                        })
                        .collect();

                    b::TypeDefBody::Interface(b::InterfaceType {
                        methods,
                        ..iface.clone()
                    })
                }
                _ => unreachable!(),
            };

        let typedef = &mut self.typedefs[i];
        typedef.typedef.body = body;
        typedef.typedef.generics = generics;
    }
}

pub struct DeclaredTypeDef<'a, 't> {
    pub typedef: b::TypeDef,
    pub type_decl_node: Option<ts::Node<'t>>,
    pub methods_idx: HashMap<&'a str, (usize, usize)>,
}

fn default_idents() -> HashMap<String, b::TypeBody> {
    HashMap::from([
        ("void".to_string(), b::TypeBody::Void),
        ("never".to_string(), b::TypeBody::Never),
        ("bool".to_string(), b::TypeBody::Bool),
        ("i8".to_string(), b::TypeBody::I8),
        ("i16".to_string(), b::TypeBody::I16),
        ("i32".to_string(), b::TypeBody::I32),
        ("i64".to_string(), b::TypeBody::I64),
        ("u8".to_string(), b::TypeBody::U8),
        ("u16".to_string(), b::TypeBody::U16),
        ("u32".to_string(), b::TypeBody::U32),
        ("u64".to_string(), b::TypeBody::U64),
        ("usize".to_string(), b::TypeBody::USize),
        ("f32".to_string(), b::TypeBody::F32),
        ("f64".to_string(), b::TypeBody::F64),
        ("str".to_string(), b::TypeBody::String),
        ("Ptr".to_string(), b::TypeBody::Ptr(None)),
    ])
}
