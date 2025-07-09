use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::{cmp, fmt};

use derive_more::{Debug, Display, From};
use derive_new::new;
use itertools::Itertools;
use tree_sitter as ts;

use super::instr::*;
use super::ty::*;
use super::value::*;
use crate::utils::{self, SortedMap};

#[derive(Debug, Clone, new)]
pub struct Module {
    pub idx:      usize,
    #[new(default)]
    pub values:   Vec<Value>,
    #[new(default)]
    pub typedefs: Vec<TypeDef>,
    #[new(default)]
    pub globals:  Vec<Global>,
    #[new(default)]
    pub funcs:    Vec<Func>,
    pub sources:  HashSet<Source>,
}
impl Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "module {}:", self.idx)?;

        for (i, value) in self.values.iter().enumerate() {
            write!(f, "\n    v{i}: {}", &value.ty)?;
            if let Some(redirects_to) = &value.redirects_to {
                write!(f, " = v{redirects_to}")?;
            } else if value.same_type_of.len() > 0 {
                write!(
                    f,
                    ", type of {}",
                    value
                        .same_type_of
                        .iter()
                        .map(|v| format!("v{v}"))
                        .join(" | ")
                )?;
            }
            write!(f, " {}", value.loc)?;
        }

        for (i, typedef) in self.typedefs.iter().enumerate() {
            write!(f, "\n    type {i} {} ", &typedef.loc)?;
            if typedef.generics.len() > 0 {
                write!(f, "(generics")?;
                for idx in &typedef.generics {
                    write!(f, " {}", idx)?;
                }
                write!(f, ") ")?;
            }

            match &typedef.body {
                TypeDefBody::Record(v) => {
                    write!(f, "record:")?;
                    if v.ifaces.len() > 0 {
                        write!(f, "\n        implements")?;
                        for iface in &v.ifaces {
                            write!(f, " {}", iface)?;
                        }
                    }
                    for (name, field) in &v.fields {
                        write!(f, "\n        {name}: {field}")?;
                    }
                    for (name, method) in &v.methods {
                        write!(f, "\n        {name}(): {method}")?;
                    }
                }
                TypeDefBody::Interface(v) => {
                    write!(f, "interface:")?;
                    for (name, method) in &v.methods {
                        write!(f, "\n        {name}(): {method}")?;
                    }
                }
            }
        }

        for (i, global) in self.globals.iter().enumerate() {
            write!(f, "\n    global {i} {} -> v{}", global.loc, global.value)?;
            if global.body.len() > 0 {
                write!(f, ":\n{}", utils::indented(8, &global.body))?;
            }
        }

        for (i, func) in self.funcs.iter().enumerate() {
            write!(f, "\n    func {i} {}", func.loc)?;

            if func.params.len() > 0 {
                write!(f, " (params")?;
                for v in &func.params {
                    write!(f, " v{v}")?;
                }
                write!(f, ")")?;
            }

            write!(f, " -> v{}", &func.ret)?;

            if let Some((mod_idx, ty_idx, name)) = &func.method {
                write!(f, " (method {mod_idx}-{ty_idx} .{name})")?;
            }

            if let Some(Extern { name }) = &func.extrn {
                write!(f, " (extern {})", utils::encode_string_lit(name))?;
            }

            if func.body.len() > 0 {
                write!(f, ":\n{}", utils::indented(8, &func.body))?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TypeDef {
    pub name:     String,
    pub body:     TypeDefBody,
    /// List of index of the (not yet implemented) module's list of generics
    pub generics: Vec<usize>,
    pub loc:      Loc,
}
impl TypeDef {
    pub fn get_ifaces(&self) -> Vec<Cow<'_, InterfaceImpl>> {
        match &self.body {
            TypeDefBody::Record(rec) => rec.ifaces.iter().map(Cow::Borrowed).collect(),
            _ => vec![],
        }
    }

    pub fn get_method<'a>(&'a self, name: &str) -> Option<&'a Method> {
        if self.generics.len() > 0 {
            todo!()
        }
        match &self.body {
            TypeDefBody::Record(rec) => rec.methods.get(name),
            TypeDefBody::Interface(iface) => iface.methods.get(name),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Global {
    pub name: String,
    pub value: ValueIdx,
    pub body: Vec<Instr>,
    pub is_entry_point: bool,
    pub loc: Loc,
}

#[derive(Debug, Clone)]
pub struct Func {
    pub name:    String,
    pub params:  Vec<ValueIdx>,
    pub ret:     ValueIdx,
    pub body:    Vec<Instr>,
    pub method:  Option<(usize, usize, String)>,
    pub extrn:   Option<Extern>,
    pub is_virt: bool,
    pub loc:     Loc,
}

#[derive(Debug, Clone, From)]
pub enum TypeDefBody {
    Record(RecordType),
    Interface(InterfaceType),
}

#[derive(Debug, Clone)]
pub struct RecordType {
    pub ifaces:  HashSet<InterfaceImpl>,
    pub fields:  SortedMap<String, RecordField>,
    pub methods: SortedMap<String, Method>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Display)]
pub enum InterfaceImpl {
    #[display("{_0}")]
    TypeRef(TypeRef),
    #[display("{_0} of {}", _1.iter().join(", "))]
    GenericInstance(TypeRef, Vec<Type>),
}
impl InterfaceImpl {
    pub fn type_ref(&self) -> TypeRef {
        match self {
            Self::TypeRef(t) => *t,
            Self::GenericInstance(t, _) => *t,
        }
    }

    pub fn mod_idx(&self) -> usize {
        self.type_ref().mod_idx
    }

    pub fn ty_idx(&self) -> usize {
        self.type_ref().idx
    }

    pub fn get_method<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<(&'a Method, HashMap<GenericRef, Cow<'a, Type>>)> {
        let (typedef, method) = match self {
            Self::TypeRef(t) | Self::GenericInstance(t, _) => {
                let typedef = modules.get(t.mod_idx)?.typedefs.get(t.idx)?;

                let method = match &typedef.body {
                    TypeDefBody::Record(rec) => rec.methods.get(name)?,
                    TypeDefBody::Interface(iface) => iface.methods.get(name)?,
                };

                (typedef, method)
            }
        };

        let generics = match self {
            Self::GenericInstance(t, args) => typedef
                .generics
                .iter()
                .enumerate()
                .map(|(i, x)| (GenericRef::new(t.mod_idx, *x), Cow::Borrowed(&args[i])))
                .collect(),
            _ => HashMap::new(),
        };

        Some((method, generics))
    }
}

#[derive(Debug, Clone)]
pub struct InterfaceType {
    pub methods: SortedMap<String, Method>,
}

#[derive(Debug, Clone, Display, new)]
#[display("{ty} {loc}")]
pub struct RecordField {
    pub name: NameWithLoc,
    pub ty:   Type,
    pub loc:  Loc,
}

#[derive(Debug, Clone, Display, new)]
#[display("({}, {}) {loc}", func_ref.0, func_ref.1)]
pub struct Method {
    pub name:     NameWithLoc,
    pub func_ref: (usize, usize),
    pub loc:      Loc,
}

#[derive(Debug, Clone, new)]
pub struct NameWithLoc {
    pub value: String,
    pub loc:   Loc,
}

#[derive(Debug, Clone)]
pub struct Extern {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, new)]
pub struct Source {
    pub path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
#[display(":{start_line}:{start_col}-{end_line}:{end_col}")]
#[debug(":{start_line}:{start_col}-{end_line}:{end_col}")]
pub struct Loc {
    pub source_idx: usize,
    pub start_line: usize,
    pub start_col:  usize,
    pub start_byte: usize,
    pub end_line:   usize,
    pub end_col:    usize,
    pub end_byte:   usize,
}
impl Loc {
    pub fn from_node(source: usize, node: &ts::Node) -> Loc {
        let start_pos = node.start_position();
        let end_pos = node.end_position();
        Loc {
            source_idx: source,
            start_line: start_pos.row + 1,
            start_col:  start_pos.column + 1,
            start_byte: node.start_byte(),
            end_line:   end_pos.row + 1,
            end_col:    end_pos.column + 1,
            end_byte:   node.end_byte(),
        }
    }
    pub fn merge(&self, other: &Loc) -> Loc {
        assert!(self.source_idx == other.source_idx);
        Loc {
            source_idx: self.source_idx,
            start_byte: cmp::min(self.start_byte, other.start_byte),
            start_line: cmp::min(self.start_line, other.start_line),
            start_col:  cmp::min(self.start_col, other.start_col),
            end_byte:   cmp::max(self.end_byte, other.end_byte),
            end_line:   cmp::max(self.end_line, other.end_line),
            end_col:    cmp::max(self.end_col, other.end_col),
        }
    }
}
