use std::collections::HashMap;
use std::fmt::{self, Display};

use derive_ctor::ctor;
use derive_more::derive::Debug;
use itertools::enumerate;

use super::{BlockIdx, Loc, Type, ValueIdx};
use crate::utils;

#[derive(Debug, Clone)]
pub enum InstrBody {
    GetGlobal(usize, usize),
    GetFunc(usize, usize),
    GetProperty(ValueIdx, String),
    GetField(ValueIdx, String),
    GetMethod(ValueIdx, String),
    CreateBool(bool),
    CreateNumber(String),
    CreateString(String),
    CreateUninitializedString(ValueIdx),
    CreateArray(Vec<ValueIdx>),
    CreateRecord(utils::SortedMap<String, ValueIdx>),

    Add(ValueIdx, ValueIdx),
    Sub(ValueIdx, ValueIdx),
    Mul(ValueIdx, ValueIdx),
    Div(ValueIdx, ValueIdx),
    Mod(ValueIdx, ValueIdx),

    Eq(ValueIdx, ValueIdx),
    Neq(ValueIdx, ValueIdx),
    Gt(ValueIdx, ValueIdx),
    Gte(ValueIdx, ValueIdx),
    Lt(ValueIdx, ValueIdx),
    Lte(ValueIdx, ValueIdx),
    Not(ValueIdx),

    Call(usize, usize, Vec<ValueIdx>),
    IndirectCall(ValueIdx, Vec<ValueIdx>),

    If(ValueIdx, BlockIdx, BlockIdx),
    Loop(Vec<(ValueIdx, ValueIdx)>, BlockIdx),
    Break(Option<ValueIdx>),
    Continue(Vec<ValueIdx>),

    StrLen(ValueIdx),
    StrPtr(ValueIdx),
    StrFromPtr(ValueIdx, ValueIdx),
    StrCopy(ValueIdx, ValueIdx, Option<ValueIdx>),

    ArrayLen(ValueIdx),
    ArrayIndex(ValueIdx, ValueIdx),

    PtrOffset(ValueIdx, ValueIdx),
    PtrSet(ValueIdx, ValueIdx),

    Type(ValueIdx, Type),
    Dispatch(ValueIdx, usize, usize),

    TypeName(ValueIdx),

    CompileError,
}

impl InstrBody {
    pub fn remap_values(&mut self, remap: &HashMap<ValueIdx, ValueIdx>) {
        let replace = |v: &mut ValueIdx| {
            if let Some(&new) = remap.get(v) {
                *v = new;
            }
        };
        match self {
            InstrBody::Call(_, _, args) => {
                for v in args {
                    replace(v);
                }
            }
            InstrBody::IndirectCall(func, args) => {
                replace(func);
                for v in args {
                    replace(v);
                }
            }
            InstrBody::TypeName(v) => replace(v),
            InstrBody::Break(Some(v)) => replace(v),
            InstrBody::Break(None) => {}
            InstrBody::Continue(vs) => {
                for v in vs {
                    replace(v);
                }
            }
            InstrBody::Not(v) => replace(v),
            InstrBody::Add(a, b)
            | InstrBody::Sub(a, b)
            | InstrBody::Mul(a, b)
            | InstrBody::Div(a, b)
            | InstrBody::Mod(a, b) => {
                replace(a);
                replace(b);
            }
            InstrBody::Eq(a, b)
            | InstrBody::Neq(a, b)
            | InstrBody::Lt(a, b)
            | InstrBody::Lte(a, b)
            | InstrBody::Gt(a, b)
            | InstrBody::Gte(a, b) => {
                replace(a);
                replace(b);
            }
            InstrBody::StrPtr(v) | InstrBody::StrLen(v) | InstrBody::ArrayLen(v) => {
                replace(v);
            }
            InstrBody::StrFromPtr(a, b)
            | InstrBody::PtrOffset(a, b)
            | InstrBody::PtrSet(a, b)
            | InstrBody::ArrayIndex(a, b) => {
                replace(a);
                replace(b);
            }
            InstrBody::StrCopy(src, dst, offset) => {
                replace(src);
                replace(dst);
                if let Some(v) = offset {
                    replace(v);
                }
            }
            InstrBody::CreateUninitializedString(v) => replace(v),
            InstrBody::GetField(v, _)
            | InstrBody::GetProperty(v, _)
            | InstrBody::GetMethod(v, _)
            | InstrBody::Dispatch(v, _, _)
            | InstrBody::Type(v, _) => replace(v),
            InstrBody::GetFunc(..)
            | InstrBody::CreateNumber(_)
            | InstrBody::CreateBool(_)
            | InstrBody::CreateString(_)
            | InstrBody::CreateRecord(_)
            | InstrBody::GetGlobal(..)
            | InstrBody::CompileError => {}
            InstrBody::If(cond, _, _) => {
                replace(cond);
            }
            InstrBody::Loop(inits, _) => {
                for (loop_var, init_val) in inits {
                    replace(loop_var);
                    replace(init_val);
                }
            }
            InstrBody::CreateArray(values) => {
                for v in values {
                    replace(v);
                }
            }
        }
    }
}

impl Display for InstrBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstrBody::GetGlobal(mod_idx, global_idx) => {
                write!(f, "get_global {mod_idx}-{global_idx}")?
            }
            InstrBody::GetFunc(mod_idx, func_idx) => {
                write!(f, "get_func {mod_idx}-{func_idx}")?
            }
            InstrBody::GetProperty(v, prop) => write!(f, "get_property v{v} .{prop}")?,
            InstrBody::GetField(v, field) => write!(f, "get_field v{v} .{field}")?,
            InstrBody::GetMethod(v, field) => write!(f, "get_method v{v} .{field}")?,
            InstrBody::CreateBool(v) => write!(f, "create_bool {v}")?,
            InstrBody::CreateNumber(v) => write!(f, "create_number {v}")?,
            InstrBody::CreateString(v) => {
                write!(f, "create_string {}", utils::encode_string_lit(v))?
            }
            InstrBody::CreateUninitializedString(len) => {
                write!(f, "create_uninitialized_string {len}")?
            }
            InstrBody::CreateArray(vs) => {
                write!(f, "create_array")?;
                for v in vs {
                    write!(f, " v{v}")?;
                }
            }
            InstrBody::CreateRecord(fields) => {
                write!(f, "create_record")?;
                for (name, v) in fields {
                    write!(f, " .{name}=v{v}")?;
                }
            }
            InstrBody::Add(a, b) => write!(f, "add v{a} v{b}")?,
            InstrBody::Sub(a, b) => write!(f, "sub v{a} v{b}")?,
            InstrBody::Mul(a, b) => write!(f, "mul v{a} v{b}")?,
            InstrBody::Div(a, b) => write!(f, "div v{a} v{b}")?,
            InstrBody::Mod(a, b) => write!(f, "mod v{a} v{b}")?,
            InstrBody::Eq(a, b) => write!(f, "eq v{a} v{b}")?,
            InstrBody::Neq(a, b) => write!(f, "neq v{a} v{b}")?,
            InstrBody::Gt(a, b) => write!(f, "gt v{a} v{b}")?,
            InstrBody::Gte(a, b) => write!(f, "gte v{a} v{b}")?,
            InstrBody::Lt(a, b) => write!(f, "lt v{a} v{b}")?,
            InstrBody::Lte(a, b) => write!(f, "lte v{a} v{b}")?,
            InstrBody::Not(v) => write!(f, "not v{v}")?,
            InstrBody::Call(mod_idx, func_idx, args) => {
                write!(f, "call {mod_idx}-{func_idx}")?;
                for arg in args {
                    write!(f, " v{arg}")?;
                }
            }
            InstrBody::IndirectCall(v, args) => {
                write!(f, "indirect_call v{v}")?;
                for arg in args {
                    write!(f, " v{arg}")?;
                }
            }
            InstrBody::If(v, then_block, else_block) => {
                write!(f, "if v{v} then block:{then_block} else block:{else_block}")?;
            }
            InstrBody::Loop(inputs, body_block) => {
                write!(f, "loop")?;
                for (v, initial_v) in inputs {
                    write!(f, " v{v}=v{initial_v}")?;
                }
                write!(f, " block:{body_block}")?;
            }
            InstrBody::Break(v) => {
                write!(f, "break")?;
                if let Some(v) = v {
                    write!(f, " v{v}")?;
                }
            }
            InstrBody::Continue(vs) => {
                write!(f, "continue")?;
                for v in vs {
                    write!(f, " v{v}")?;
                }
            }
            InstrBody::StrLen(v) => write!(f, "str_len v{v}")?,
            InstrBody::StrPtr(v) => write!(f, "str_ptr v{v}")?,
            InstrBody::StrFromPtr(ptr, len) => write!(f, "str_from_ptr v{ptr} v{len}")?,
            InstrBody::StrCopy(src, dst, offset) => {
                write!(f, "str_copy v{src} v{dst}")?;
                if let Some(offset) = offset {
                    write!(f, "+v{offset}")?;
                }
            }
            InstrBody::ArrayLen(v) => write!(f, "array_len v{v}")?,
            InstrBody::ArrayIndex(v, idx) => write!(f, "array_index v{v} v{idx}")?,
            InstrBody::PtrOffset(ptr, offset) => {
                write!(f, "ptr_offset v{ptr} v{offset}")?
            }
            InstrBody::PtrSet(ptr, value) => write!(f, "ptr_set v{ptr} v{value}")?,
            InstrBody::Type(v, ty) => write!(f, "type v{v} {ty}")?,
            InstrBody::Dispatch(v, mod_idx, ty_idx) => {
                write!(f, "dispatch v{v} {mod_idx}-{ty_idx}")?
            }
            InstrBody::TypeName(v) => write!(f, "type_name v{v}")?,
            InstrBody::CompileError => write!(f, "compile_error")?,
        }
        Ok(())
    }
}

#[derive(Debug, Clone, ctor)]
pub struct Instr {
    pub body:    InstrBody,
    pub loc:     Option<Loc>,
    #[ctor(default)]
    pub results: Vec<ValueIdx>,
}
impl Instr {
    pub fn get_global(
        mod_idx: usize,
        global_idx: usize,
        res: ValueIdx,
        loc: Option<Loc>,
    ) -> Self {
        Self::new(InstrBody::GetGlobal(mod_idx, global_idx), loc).with_results([res])
    }

    pub fn create_number(v: String, res: ValueIdx, loc: Option<Loc>) -> Self {
        Self::new(InstrBody::CreateNumber(v), loc).with_results([res])
    }

    pub fn add(a: ValueIdx, b: ValueIdx, res: ValueIdx, loc: Option<Loc>) -> Self {
        Self::new(InstrBody::Add(a, b), loc).with_results([res])
    }

    pub fn gte(a: ValueIdx, b: ValueIdx, res: ValueIdx, loc: Option<Loc>) -> Self {
        Self::new(InstrBody::Gte(a, b), loc).with_results([res])
    }

    pub fn lt(a: ValueIdx, b: ValueIdx, res: ValueIdx, loc: Option<Loc>) -> Self {
        Self::new(InstrBody::Lt(a, b), loc).with_results([res])
    }

    pub fn call(
        mod_idx: usize,
        func_idx: usize,
        args: impl IntoIterator<Item = ValueIdx>,
        res: ValueIdx,
        loc: Option<Loc>,
    ) -> Self {
        Self::new(
            InstrBody::Call(mod_idx, func_idx, args.into_iter().collect()),
            loc,
        )
        .with_results([res])
    }

    pub fn if_(
        cond: ValueIdx,
        then_block: BlockIdx,
        else_block: BlockIdx,
        res: Option<ValueIdx>,
        loc: Option<Loc>,
    ) -> Self {
        Self::new(InstrBody::If(cond, then_block, else_block), loc).with_results(res)
    }

    pub fn loop_(
        inputs: Vec<(ValueIdx, ValueIdx)>,
        body_block: BlockIdx,
        res: Option<ValueIdx>,
        loc: Option<Loc>,
    ) -> Self {
        Self::new(InstrBody::Loop(inputs, body_block), loc).with_results(res)
    }

    pub fn break_(v: Option<ValueIdx>, loc: Option<Loc>) -> Self {
        Self::new(InstrBody::Break(v), loc)
    }

    pub fn continue_(vs: Vec<ValueIdx>, loc: Option<Loc>) -> Self {
        Self::new(InstrBody::Continue(vs), loc)
    }

    pub fn array_len(v: ValueIdx, res: ValueIdx, loc: Option<Loc>) -> Self {
        Self::new(InstrBody::ArrayLen(v), loc).with_results([res])
    }

    pub fn array_index(
        v: ValueIdx,
        idx: ValueIdx,
        res: ValueIdx,
        loc: Option<Loc>,
    ) -> Self {
        Self::new(InstrBody::ArrayIndex(v, idx), loc).with_results([res])
    }

    pub fn type_name(v: ValueIdx, res: ValueIdx, loc: Option<Loc>) -> Self {
        Self::new(InstrBody::TypeName(v), loc).with_results([res])
    }

    pub fn with_results(mut self, results: impl IntoIterator<Item = ValueIdx>) -> Self {
        self.results = results.into_iter().collect();
        self
    }
}
impl Display for Instr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.results.len() > 0 {
            write!(f, "{} = ", utils::join(", ", &self.results))?;
        }
        for (i, line) in enumerate(self.body.to_string().split('\n')) {
            if i != 0 {
                write!(f, "\n")?;
            }
            write!(f, "{line}")?;
            if i == 0 {
                if let Some(loc) = &self.loc {
                    write!(f, " {}", &loc)?;
                }
            }
        }
        Ok(())
    }
}
