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
    pub modules: &'a [b::Module],
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
            b::TypeBody::Void => write!(s, "v").unwrap(),
            b::TypeBody::Bool => write!(s, "b").unwrap(),
            b::TypeBody::I8 => write!(s, "a").unwrap(),
            b::TypeBody::I16 => write!(s, "s").unwrap(),
            b::TypeBody::I32 => write!(s, "i").unwrap(),
            b::TypeBody::I64 => write!(s, "x").unwrap(),
            b::TypeBody::U8 => write!(s, "h").unwrap(),
            b::TypeBody::U16 => write!(s, "t").unwrap(),
            b::TypeBody::U32 => write!(s, "j").unwrap(),
            b::TypeBody::U64 => write!(s, "y").unwrap(),
            b::TypeBody::USize => self.write_name(
                s,
                &b::Name::from_ident("usize", b::NameIdentKind::Type, None),
            ),
            b::TypeBody::F32 => write!(s, "f").unwrap(),
            b::TypeBody::F64 => write!(s, "d").unwrap(),
            b::TypeBody::String => self
                .write_name(s, &b::Name::from_ident("str", b::NameIdentKind::Type, None)),
            b::TypeBody::Array(ty) => {
                let name = b::Name::from_ident("array", b::NameIdentKind::Type, None)
                    .with_type_params([ty.as_ref().clone()], None);
                self.write_name(s, &name);
            }
            b::TypeBody::Ptr(None) => write!(s, "Pv").unwrap(),
            b::TypeBody::Ptr(Some(ty)) => {
                write!(s, "P").unwrap();
                self.write_type(s, ty);
            }
            b::TypeBody::Func(func) => {
                write!(s, "F").unwrap();
                for ty in &func.params {
                    self.write_type(s, ty);
                }
                write!(s, "E").unwrap();
            }
            b::TypeBody::TypeRef(ty_ref) => {
                let ty = &self.modules[ty_ref.mod_idx].typedefs[ty_ref.idx];
                self.write_name(s, &ty.name);
            }
            b::TypeBody::Never
            | b::TypeBody::AnyOpaque
            | b::TypeBody::AnyNumber
            | b::TypeBody::AnySignedNumber
            | b::TypeBody::AnyFloat
            | b::TypeBody::Inferred(_)
            | b::TypeBody::TypeVar(_) => panic!("cannot mangle type `{ty}`"),
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
