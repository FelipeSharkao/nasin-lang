use std::collections::{HashMap, HashSet};

use derive_new::new;
use itertools::Itertools;
use tree_sitter as ts;

use crate::utils::{IntoItem, SortedMap, TreeSitterUtils};
use crate::{bytecode as b, context, errors};

#[derive(new)]
pub struct TypeParser<'a, 't> {
    #[new(default)]
    pub typedefs: Vec<DeclaredTypeDef<'a, 't>>,
    #[new(value = "default_idents()")]
    pub idents: HashMap<String, b::TypeBody>,
    ctx: &'a context::BuildContext,
    src_idx: usize,
    mod_idx: usize,
}

impl<'a, 't> TypeParser<'a, 't> {
    pub fn finish(self) -> Vec<b::TypeDef> {
        self.typedefs
            .iter()
            .map(|x| self.finish_typedef(x))
            .collect()
    }
    pub fn parse_type_expr(&self, node: ts::Node<'t>) -> b::Type {
        let node = node.of_kind("type_expr").child(0).unwrap();

        let body = match node.kind() {
            "ident" => {
                let ident = node.get_text(&self.ctx.source(self.src_idx).content().text);
                match self.idents.get(ident) {
                    Some(body) => body.clone(),
                    None => {
                        self.ctx.push_error(errors::Error::new(
                            errors::TypeNotFound::new(ident.to_string()).into(),
                            b::Loc::from_node(self.src_idx, &node),
                        ));
                        b::TypeBody::unknown()
                    }
                }
            }
            "array_type" => {
                let item_ty = self.parse_type_expr(node.required_field("item_type"));
                let len = node.field("length").map(|n| {
                    n.get_text(&self.ctx.source(self.src_idx).content().text)
                        .parse::<usize>()
                        .expect("Cannot cast length to integer")
                });
                b::TypeBody::Array(b::ArrayType::new(
                    item_ty.into(),
                    len.map(|x| x as u32),
                ))
            }
            "generic_type" => {
                let name = node
                    .required_field("name")
                    .of_kind("ident")
                    .get_text(&self.ctx.source(self.src_idx).content().text);

                let args = node
                    .iter_field("args")
                    .map(|arg_node| self.parse_type_expr(arg_node))
                    .collect_vec();

                match name {
                    "Ptr" => {
                        // TODO: Better error handling
                        assert!(args.len() == 1, "Ptr accepts only one parameter");
                        b::TypeBody::Ptr(args.into_item(0).unwrap().into())
                    }
                    _ => panic!("unhandled generic type: `{name}`"),
                }
            }
            k => panic!("Unhandled type node `{k}`"),
        };
        b::Type::new(body, Some(b::Loc::from_node(self.src_idx, &node)))
    }
    pub fn parse_type_decl(
        &mut self,
        name: String,
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

        let value = b::TypeDef {
            name: name,
            body,
            loc: b::Loc::from_node(self.src_idx, &node),
        };
        self.idents.insert(
            value.name.clone(),
            b::TypeRef::new(self.mod_idx, self.typedefs.len()).into(),
        );
        self.typedefs.push(DeclaredTypeDef {
            typedef: value,
            type_decl_node: Some(node),
            methods_idx,
        });
    }

    fn finish_typedef(&self, x: &DeclaredTypeDef) -> b::TypeDef {
        let Some(node) = x.type_decl_node else {
            return x.typedef.clone();
        };
        let body_node = node.required_field("body");
        let body =
            match (body_node.kind(), &x.typedef.body) {
                ("record_type", b::TypeDefBody::Record(rec)) => {
                    let fields = body_node
                        .iter_field("fields")
                        .map(|field_node| {
                            let name_node = field_node.required_field("name");
                            let name = name_node
                                .get_text(&self.ctx.source(self.src_idx).content().text)
                                .to_string();
                            let record_field = b::RecordField::new(
                                b::NameWithLoc::new(
                                    name.clone(),
                                    b::Loc::from_node(self.src_idx, &name_node),
                                ),
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
                                .get_text(&self.ctx.source(self.src_idx).content().text)
                                .to_string();
                            let func_ref = x.methods_idx.get(&name as &str).expect(
                                "index of method's function should already be known",
                            );

                            let method = b::Method::new(
                                b::NameWithLoc::new(
                                    name.clone(),
                                    b::Loc::from_node(self.src_idx, &name_node),
                                ),
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
                                    errors::TypeNotInterface::new(ty).into(),
                                    b::Loc::from_node(self.src_idx, &node),
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
                                .get_text(&self.ctx.source(self.src_idx).content().text)
                                .to_string();
                            let func_ref = x.methods_idx.get(&name as &str).expect(
                                "index of method's function should already be known",
                            );

                            let method = b::Method::new(
                                b::NameWithLoc::new(
                                    name.clone(),
                                    b::Loc::from_node(self.src_idx, &name_node),
                                ),
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
        b::TypeDef {
            body,
            ..x.typedef.clone()
        }
    }
}

pub struct DeclaredTypeDef<'a, 't> {
    pub typedef:    b::TypeDef,
    type_decl_node: Option<ts::Node<'t>>,
    methods_idx:    HashMap<&'a str, (usize, usize)>,
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
        (
            "str".to_string(),
            b::TypeBody::String(b::StringType { len: None }),
        ),
    ])
}
