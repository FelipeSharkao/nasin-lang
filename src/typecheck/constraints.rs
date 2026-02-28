use derive_new::new;

use crate::bytecode as b;
use crate::utils::{number_enum, SortedMap};

number_enum!(pub ConstraintPriority: u8 {
    NoType = 0,
    DerivedInferredType = 1,
    DerivedDefinedType = 2,
    DefinedType = 3,
});

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstraintKind {
    Is(b::Type),
    TypeOf(b::ValueIdx),
    Array(b::ValueIdx),
    ArrayElem(b::ValueIdx),
    Ptr(b::ValueIdx),
    Deref(b::ValueIdx),
    ReturnOf(b::ValueIdx),
    ParameterOf(b::ValueIdx, usize),
    IsProperty(b::ValueIdx, String),
    Members(SortedMap<String, b::ValueIdx>),
    HasProperty(String, b::ValueIdx),
    GetFunc(usize, usize),
    Func(Vec<b::ValueIdx>, b::ValueIdx),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, new)]
pub struct Constraint {
    pub kind: ConstraintKind,
    pub loc:  Option<b::Loc>,
}
impl Constraint {
    pub fn priority(&self) -> ConstraintPriority {
        match &self.kind {
            ConstraintKind::Is(..) => ConstraintPriority::DefinedType,
            ConstraintKind::TypeOf(..)
            | ConstraintKind::Array(..)
            | ConstraintKind::ArrayElem(..)
            | ConstraintKind::Ptr(..)
            | ConstraintKind::Deref(..)
            | ConstraintKind::ReturnOf(..)
            | ConstraintKind::ParameterOf(..)
            | ConstraintKind::GetFunc(..)
            | ConstraintKind::IsProperty(..) => ConstraintPriority::DerivedDefinedType,
            ConstraintKind::Members(..)
            | ConstraintKind::HasProperty(..)
            | ConstraintKind::Func(..) => ConstraintPriority::DerivedInferredType,
        }
    }
}
