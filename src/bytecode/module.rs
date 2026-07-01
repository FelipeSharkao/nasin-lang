use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use derive_ctor::ctor;
use derive_more::{Debug, Display, From};
use itertools::izip;
use nasin_macros::NumberEnum;
use tree_sitter as ts;

use super::Printer;
use super::instr::*;
use super::name::*;
use super::ty::*;
use super::value::*;
use crate::config::BuildConfig;
use crate::utils::SortedMap;

#[derive(Debug, Clone, ctor)]
pub struct ImplDecl {
    pub iface: TypeRefKey,
    pub iface_args: Vec<TypeBody>,
    pub type_args_constraints: Option<Vec<TypeBody>>,
    pub loc: Loc,
    #[ctor(default)]
    pub used_type_args: HashSet<Vec<TypeBody>>,
}

impl ImplDecl {
    pub fn constraints_satisfied(&self, args: &[TypeBody], modules: &[Module]) -> bool {
        let Some(constraints) = self.type_args_constraints.as_ref() else {
            return true;
        };
        if args.len() != constraints.len() {
            return false;
        }
        izip!(args, constraints).all(|(arg, constraint)| {
            arg.merge(constraint, Variance::Covariant, modules)
                .is_some()
        })
    }
}

pub const BUILTINS_MODULE_IDX: usize = 0;

pub type BlockIdx = usize;

#[derive(Debug, Clone, Default, ctor)]
pub struct Block {
    pub body: Vec<Instr>,
    pub loc:  Option<Loc>,
}

impl Block {
    pub fn extend(&mut self, instrs: impl IntoIterator<Item = Instr>) {
        let old_len = self.body.len();
        self.body.extend(instrs);
        for instr in &self.body[old_len..] {
            match (self.loc, instr.loc) {
                (None, Some(new_loc)) => self.loc = Some(new_loc),
                (Some(old_loc), Some(new_loc)) => {
                    self.loc = Some(old_loc.merge(&new_loc))
                }
                _ => {}
            }
        }
    }
}

impl FromIterator<Instr> for Block {
    fn from_iter<T: IntoIterator<Item = Instr>>(iter: T) -> Self {
        let mut block = Self::default();
        block.extend(iter);
        block
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
        block_remap: &mut HashMap<BlockIdx, BlockIdx>,
    ) -> BlockIdx {
        let new_block_idx = self.add_block([]);
        block_remap.insert(block_idx, new_block_idx);

        let mut new_body = self.blocks[block_idx].body.clone();

        for instr in &mut new_body {
            match &mut instr.body {
                InstrBody::If(_, then_block, else_block) => {
                    *then_block =
                        self.clone_block_tree(*then_block, transformer, block_remap);
                    *else_block =
                        self.clone_block_tree(*else_block, transformer, block_remap);
                }
                InstrBody::Loop(_, body_block) => {
                    *body_block =
                        self.clone_block_tree(*body_block, transformer, block_remap);
                }
                InstrBody::Break(block, _) => {
                    if let Some(new_block) = block_remap.get(block) {
                        *block = *new_block;
                    }
                }
                _ => {}
            }

            transformer.remap_instr(self, instr);
        }

        self.blocks[new_block_idx].body = new_body;
        new_block_idx
    }

    pub fn clone_block_tree_rec(
        &mut self,
        block_idx: BlockIdx,
        transformer: &mut impl BlockTransformer,
        block_remap: &mut HashMap<BlockIdx, BlockIdx>,
    ) -> BlockIdx {
        let new_block_idx = self.add_block([]);
        block_remap.insert(block_idx, new_block_idx);

        let mut new_body = self.blocks[block_idx].body.clone();

        for instr in &mut new_body {
            match &mut instr.body {
                InstrBody::If(_, then_block, else_block) => {
                    *then_block =
                        self.clone_block_tree(*then_block, transformer, block_remap);
                    *else_block =
                        self.clone_block_tree(*else_block, transformer, block_remap);
                }
                InstrBody::Loop(_, body_block) => {
                    *body_block =
                        self.clone_block_tree(*body_block, transformer, block_remap);
                }
                InstrBody::Break(block, _) => {
                    if let Some(new_block) = block_remap.get(block) {
                        *block = *new_block;
                    }
                }
                _ => {}
            }

            transformer.remap_instr(self, instr);
        }

        self.blocks[new_block_idx].body = new_body;
        new_block_idx
    }
}

pub trait BlockTransformer {
    fn remap_instr(&mut self, module: &mut Module, instr: &mut Instr);
}

#[derive(Debug, Clone, ctor)]
pub struct TypeDef {
    pub name:     Name,
    pub body:     TypeDefBody,
    pub loc:      Loc,
    #[ctor(iter(TypeVarIdx))]
    pub generics: Vec<TypeVarIdx>,
    #[ctor(default)]
    pub impls:    Vec<ImplDecl>,
    #[ctor(default)]
    pub methods:  SortedMap<String, Method>,
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
    pub method: Option<FuncMethodInfo>,
    pub extrn: Option<Extern>,
    pub is_entry: bool,
    pub is_virt: bool,
    pub loc: Option<Loc>,
    pub generics: Vec<TypeVarIdx>,
    /// Maps generic substitutions to the index of the instantiated func. Used to
    /// deduplicate generic instantiations
    pub generic_instantiations: HashMap<Vec<TypeBody>, usize>,
}

impl Func {
    pub fn formated_signature(
        &self,
        mod_idx: usize,
        modules: &[Module],
        cfg: &BuildConfig,
        base_module: Option<usize>,
    ) -> String {
        let params = self
            .params
            .iter()
            .map(|v| &modules[mod_idx].values[*v].ty.body);
        let ret = &modules[mod_idx].values[self.ret].ty.body;
        let mut s = String::new();
        let mut printer = Printer::new(modules, cfg);
        if let Some(base_module) = base_module {
            printer = printer.with_cur_mod_idx(base_module);
        }
        printer.write_signature(&mut s, params, ret).unwrap();
        s
    }
}

#[derive(Debug, Clone, ctor)]
pub struct FuncMethodInfo {
    pub name:       String,
    pub ty:         TypeRefKey,
    pub is_virtual: bool,
    #[ctor(default)]
    pub ty_args:    Option<Vec<Type>>,
}

#[derive(Debug, Clone, From)]
pub enum TypeDefBody {
    Record(RecordType),
    Interface,
    Builtin(BuiltinType),
}

#[derive(Debug, Clone, ctor)]
pub struct TypeVarDef {
    pub name:       Name,
    pub constraint: Option<Type>,
    pub loc:        Option<Loc>,
}

pub type TypeVarIdx = usize;

#[derive(Debug, Clone, ctor)]
pub struct RecordType {
    #[ctor(default)]
    pub fields: SortedMap<String, RecordField>,
}

#[derive(Debug, Clone, ctor)]
pub struct RecordField {
    pub ty:  Type,
    pub loc: Loc,
}

/// Builtin types. Every compilation has a module 0 (`BUILTINS_MODULE_IDX`) with TypeDefs
/// for these types so they can have methods and interfaces.
#[derive(Debug, Hash, NumberEnum)]
#[repr(usize)]
pub enum BuiltinType {
    Void,
    Never,
    Bool,
    AnyOpaque,
    AnyNumber,
    AnySignedNumber,
    AnyFloat,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    USize,
    F32,
    F64,
    String,
    Array,
    Ptr,
}

impl BuiltinType {
    pub fn is_primitive(&self) -> bool {
        matches!(self, Self::Bool | Self::Ptr) || self.is_number()
    }

    pub fn is_aggregate(&self) -> bool {
        matches!(self, Self::String | Self::Array)
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Self::AnyNumber | Self::AnySignedNumber)
            || self.is_int()
            || self.is_float()
    }

    pub fn is_int(&self) -> bool {
        self.is_sint() || self.is_uint()
    }

    pub fn is_sint(&self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    pub fn is_uint(&self) -> bool {
        matches!(
            self,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::USize
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Self::AnyFloat | Self::F32 | Self::F64)
    }

    pub fn is_not_final(&self) -> bool {
        matches!(
            self,
            Self::AnyNumber | Self::AnySignedNumber | Self::AnyFloat
        )
    }
}

#[derive(Debug, Clone, ctor)]
pub struct Method {
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
            start_byte: usize::min(self.start_byte, other.start_byte),
            start_line: usize::min(self.start_line, other.start_line),
            start_col:  usize::min(self.start_col, other.start_col),
            end_byte:   usize::max(self.end_byte, other.end_byte),
            end_line:   usize::max(self.end_line, other.end_line),
            end_col:    usize::max(self.end_col, other.end_col),
        }
    }
}
