use std::cmp::Ordering;

use derive_ctor::ctor;
use nasin_macros::NumberEnum;

use crate::bytecode as b;
use crate::utils::SortedMap;

#[derive(Debug, NumberEnum, Hash)]
#[repr(u8)]
pub enum Priority {
    NoType,
    DerivedInferred,
    DerivedDefined,
    UserDefined,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ConstraintKind {
    // The order here is kinda important, a variant declared first has less priority, tho
    // this only matters only in edge cases
    HasProperty(String, b::ValueIdx),
    Members(SortedMap<String, b::ValueIdx>),
    Func(Vec<b::ValueIdx>, b::ValueIdx),
    Array(b::ValueIdx),
    ArrayElem(b::ValueIdx),
    Ptr(b::ValueIdx),
    Deref(b::ValueIdx),
    ReturnOf(b::ValueIdx),
    ParameterOf(b::ValueIdx, usize),
    IsProperty(b::ValueIdx, String),
    GetFunc(usize, usize),
    TypeOf(b::ValueIdx, Priority, /** rigid: */ bool),
    Is(b::Type, Priority),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, ctor)]
pub struct Constraint {
    pub kind: ConstraintKind,
    pub loc:  Option<b::Loc>,
}

impl Constraint {
    pub fn priority(&self) -> Priority {
        match &self.kind {
            ConstraintKind::Is(_, priority) | ConstraintKind::TypeOf(_, priority, _) => {
                *priority
            }
            ConstraintKind::Array(..)
            | ConstraintKind::ArrayElem(..)
            | ConstraintKind::Ptr(..)
            | ConstraintKind::Deref(..)
            | ConstraintKind::ReturnOf(..)
            | ConstraintKind::ParameterOf(..)
            | ConstraintKind::GetFunc(..)
            | ConstraintKind::IsProperty(..) => Priority::DerivedDefined,
            ConstraintKind::Members(..)
            | ConstraintKind::HasProperty(..)
            | ConstraintKind::Func(..) => Priority::DerivedInferred,
        }
    }
}

impl Ord for Constraint {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&other.priority(), &self.priority())
            .then_with(|| self.loc.cmp(&other.loc))
            .then_with(|| self.kind.cmp(&other.kind))
    }
}

impl PartialOrd for Constraint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
