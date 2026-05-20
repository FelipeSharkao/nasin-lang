use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Write;

use derive_ctor::ctor;

use crate::bytecode as b;
use crate::utils::to_radix;

static PFX: &str = "_n";

/// Itanium C++ name mangling, with some modifications:
/// All names are included in the _n namespace, so that they don't clash with C++ names
/// Therefore, all names are are considered nested
#[derive(Debug, ctor)]
pub struct NameMangler<'a> {
    pub modules:        &'a [b::Module],
    #[ctor(default)]
    substitution_table: HashMap<Vec<b::NameNode>, usize>,
}

impl<'a> NameMangler<'a> {
    pub fn mangle<'b>(
        &mut self,
        name: &'b b::Name,
        signature: impl IntoIterator<Item = &'b b::Type>,
    ) -> String {
        let mut s = "_Z".to_string();
        self.write_name(&mut s, name);
        for ty in signature {
            self.write_type(&mut s, ty);
        }
        s
    }

    pub fn mangle_func(&mut self, mod_idx: usize, func_idx: usize) -> String {
        let func = &self.modules[mod_idx].funcs[func_idx];
        let signature = func
            .params
            .iter()
            .map(|param| &self.modules[mod_idx].values[*param].ty);
        self.mangle(&func.name, signature)
    }

    fn write_name(&mut self, s: &mut String, name: &b::Name) {
        let mut nodes = name.nodes.clone();
        nodes.insert(
            0,
            b::NameIdent {
                ident: PFX.into(),
                kind:  b::NameIdentKind::Module,
            }
            .into(),
        );
        let mut i = nodes
            .iter()
            .position(|node| !matches!(node, b::NameNode::Ident(_)))
            .unwrap_or(nodes.len());
        while i > 0 {
            if let Some(substitution) = self.substitution_table.get(&nodes[0..i]) {
                if i < nodes.len() {
                    s.push_str("N");
                }
                self.write_substitution(s, *substitution);
                break;
            }
            i -= 1;
        }
        // If we were able to substitute the whole name, we are done
        if i == nodes.len() {
            return;
        } else if i == 0 {
            s.push_str("N");
        }

        for (j, node) in nodes[i..].iter().enumerate() {
            match node {
                b::NameNode::Ident(ident) => {
                    write!(s, "{}{}", ident.ident.len(), ident.ident).unwrap();
                }
                b::NameNode::TypeParams(params) => {
                    write!(s, "I").unwrap();
                    for param in &params.params {
                        self.write_type(s, param);
                    }
                    write!(s, "E").unwrap();
                }
            }

            self.substitution_table
                .insert(nodes[0..=i + j].to_vec(), self.substitution_table.len());
        }
        write!(s, "E").unwrap();
    }

    fn write_type(&mut self, s: &mut String, ty: &b::Type) {
        match &ty.body {
            b::TypeBody::Func(func) => {
                write!(s, "F").unwrap();
                for ty in &func.params {
                    self.write_type(s, ty);
                }
                write!(s, "E").unwrap();
            }
            b::TypeBody::TypeRef(ty_ref) => {
                let typedefdef = ty_ref.get_typedef(self.modules);
                if let &b::TypeDefBody::Builtin(builtin) = &typedefdef.body {
                    self.write_builtin_type(s, builtin, &ty_ref.args);
                } else {
                    let mut name = Cow::Borrowed(&typedefdef.name);
                    if !ty_ref.args.is_empty() {
                        name = Cow::Owned(
                            name.with_type_params(ty_ref.args.iter().cloned(), None),
                        );
                    }
                    self.write_name(s, name.as_ref());
                }
            }
            b::TypeBody::Inferred(_) | b::TypeBody::TypeVar(_) => {
                panic!("cannot mangle type `{:?}`", &ty.body)
            }
        }
    }

    fn write_builtin_type(
        &mut self,
        s: &mut String,
        builtin: b::BuiltinType,
        args: &[b::Type],
    ) {
        match (builtin, args) {
            (b::BuiltinType::Void, []) => write!(s, "v").unwrap(),
            (b::BuiltinType::Bool, []) => write!(s, "b").unwrap(),
            (b::BuiltinType::I8, []) => write!(s, "a").unwrap(),
            (b::BuiltinType::I16, []) => write!(s, "s").unwrap(),
            (b::BuiltinType::I32, []) => write!(s, "i").unwrap(),
            (b::BuiltinType::I64, []) => write!(s, "x").unwrap(),
            (b::BuiltinType::U8, []) => write!(s, "h").unwrap(),
            (b::BuiltinType::U16, []) => write!(s, "t").unwrap(),
            (b::BuiltinType::U32, []) => write!(s, "j").unwrap(),
            (b::BuiltinType::U64, []) => write!(s, "y").unwrap(),
            (b::BuiltinType::USize, []) => self.write_name(
                s,
                &b::Name::from_ident("usize", b::NameIdentKind::Type, None),
            ),
            (b::BuiltinType::F32, []) => write!(s, "f").unwrap(),
            (b::BuiltinType::F64, []) => write!(s, "d").unwrap(),
            (b::BuiltinType::String, []) => self
                .write_name(s, &b::Name::from_ident("str", b::NameIdentKind::Type, None)),
            (b::BuiltinType::Array, [ty]) => {
                let name = b::Name::from_ident("array", b::NameIdentKind::Type, None)
                    .with_type_params([ty.clone()], None);
                self.write_name(s, &name);
            }
            (b::BuiltinType::Ptr, []) => write!(s, "Pv").unwrap(),
            (b::BuiltinType::Ptr, [ty]) => {
                write!(s, "P").unwrap();
                self.write_type(s, ty);
            }
            _ => panic!("cannot mangle type {builtin:?}"),
        }
    }

    /// Writes the substitution string, as described in the Itanium C++ ABI.
    /// S0_, S1_, ... S9_, SA_, SB_, ... SZ_, S10_, ...
    fn write_substitution(&self, s: &mut String, n: usize) {
        if n == 0 {
            write!(s, "S_").unwrap();
        } else {
            write!(s, "S{}_", to_radix(n - 1, 36)).unwrap();
        }
    }
}
