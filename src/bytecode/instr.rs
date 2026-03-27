use std::collections::HashMap;

use derive_ctor::ctor;
use derive_more::derive::Debug;

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
