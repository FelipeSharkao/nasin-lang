use std::cmp;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use derive_ctor::ctor;
use derive_more::{Debug, Display, From};
use tree_sitter as ts;

use super::instr::*;
use super::name::*;
use super::ty::*;
use super::value::*;
use crate::utils::SortedMap;

pub type BlockIdx = usize;

#[derive(Debug, Clone, ctor)]
pub struct Block {
    pub body: Vec<Instr>,
    pub loc:  Option<Loc>,
}

impl FromIterator<Instr> for Block {
    fn from_iter<T: IntoIterator<Item = Instr>>(iter: T) -> Self {
        let mut loc = None;
        let body = iter
            .into_iter()
            .map(|instr| {
                match (loc, instr.loc) {
                    (None, Some(new_loc)) => loc = Some(new_loc),
                    (Some(old_loc), Some(new_loc)) => loc = Some(old_loc.merge(&new_loc)),
                    _ => {}
                }
                instr
            })
            .collect();
        Block::new(body, loc)
    }
}

#[derive(Debug, Clone, ctor)]
pub struct Module {
    pub idx:      usize,
    pub name:     Name,
    #[ctor(default)]
    pub values:   Vec<Value>,
    #[ctor(default)]
    pub typedefs: Vec<TypeDef>,
    #[ctor(default)]
    pub typevars: Vec<TypeVarDef>,
    #[ctor(default)]
    pub globals:  Vec<Global>,
    #[ctor(default)]
    pub funcs:    Vec<Func>,
    #[ctor(default)]
    pub blocks:   Vec<Block>,
    pub sources:  HashSet<Source>,
}

impl Module {
    pub fn get_func(&self, name: &str) -> Option<(usize, &Func)> {
        self.funcs
            .iter()
            .enumerate()
            .find(|(_, f)| f.name.last_ident() == name)
    }

    pub fn add_func(&mut self, func: Func) -> usize {
        self.funcs.push(func);
        self.funcs.len() - 1
    }

    pub fn add_value(&mut self, val: Value) -> ValueIdx {
        self.values.push(val);
        self.values.len() - 1
    }

    pub fn add_block(&mut self, body: impl IntoIterator<Item = Instr>) -> BlockIdx {
        self.blocks.push(Block::from_iter(body));
        self.blocks.len() - 1
    }

    /// Deep-clone a block and all transitively referenced sub-blocks, applying
    /// a transformer to remap value indices and modify instructions. Returns
    /// the `BlockIdx` of the newly created root block.
    pub fn clone_block_tree(
        &mut self,
        block_idx: BlockIdx,
        transformer: &mut impl BlockTransformer,
    ) -> BlockIdx {
        let mut new_body = self.blocks[block_idx].body.clone();

        for instr in &mut new_body {
            match &instr.body {
                InstrBody::If(cond, then_block, else_block) => {
                    let new_then = self.clone_block_tree(*then_block, transformer);
                    let new_else = self.clone_block_tree(*else_block, transformer);
                    instr.body = InstrBody::If(*cond, new_then, new_else);
                }
                InstrBody::Loop(_, body_block) => {
                    let new_body_block = self.clone_block_tree(*body_block, transformer);
                    match &mut instr.body {
                        InstrBody::Loop(_, block) => *block = new_body_block,
                        _ => unreachable!(),
                    }
                }
                _ => {}
            }

            transformer.remap_instr(self, instr);
        }

        self.add_block(new_body)
    }
}

pub trait BlockTransformer {
    fn remap_instr(&mut self, module: &mut Module, instr: &mut Instr);
}

#[derive(Debug, Clone)]
pub struct TypeDef {
    pub name: Name,
    pub body: TypeDefBody,
    pub loc:  Loc,
}
impl TypeDef {
    pub fn get_ifaces(&self) -> Vec<(usize, usize)> {
        match &self.body {
            TypeDefBody::Record(rec) => rec.ifaces.iter().cloned().collect(),
            _ => vec![],
        }
    }

    pub fn get_method(&self, name: &str) -> Option<&Method> {
        match &self.body {
            TypeDefBody::Record(rec) => rec.methods.get(name),
            TypeDefBody::Interface(iface) => iface.methods.get(name),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Global {
    pub name:  Name,
    pub value: ValueIdx,
    pub body:  BlockIdx,
    pub loc:   Loc,
}

#[derive(Debug, Clone)]
pub struct Func {
    pub name: Name,
    pub params: Vec<ValueIdx>,
    pub ret: ValueIdx,
    pub body: BlockIdx,
    pub method: Option<(usize, usize, String)>,
    pub extrn: Option<Extern>,
    pub is_entry: bool,
    pub is_virt: bool,
    pub loc: Option<Loc>,
    pub generics: Vec<TypeVarIdx>,
    /// Maps generic substitutions to the index of the instantiated func. Used to
    /// deduplicate generic instantiations
    pub generic_instantiations: HashMap<Vec<TypeBody>, usize>,
}

#[derive(Debug, Clone, From)]
pub enum TypeDefBody {
    Record(RecordType),
    Interface(InterfaceType),
}

#[derive(Debug, Clone)]
pub struct TypeVarDef {
    pub name: Name,
    pub loc:  Loc,
}

pub type TypeVarIdx = usize;

#[derive(Debug, Clone)]
pub struct RecordType {
    pub ifaces:  HashSet<(usize, usize)>,
    pub fields:  SortedMap<String, RecordField>,
    pub methods: SortedMap<String, Method>,
}

#[derive(Debug, Clone)]
pub struct InterfaceType {
    pub methods: SortedMap<String, Method>,
}

#[derive(Debug, Clone, ctor)]
pub struct RecordField {
    pub name: String,
    pub ty:   Type,
    pub loc:  Loc,
}

#[derive(Debug, Clone, ctor)]
pub struct Method {
    pub name:     String,
    pub func_ref: (usize, usize),
    pub loc:      Loc,
}

#[derive(Debug, Clone)]
pub struct Extern {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, ctor)]
pub struct Source {
    pub path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, Default)]
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
