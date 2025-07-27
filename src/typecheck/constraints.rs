use crate::bytecode as b;
use crate::utils::{number_enum, SortedMap};

number_enum!(pub ConstraintPriority: u8 {
    NoType = 0,
    DerivedInferredType = 1,
    DerivedDefinedType = 2,
    DefinedType = 3,
});

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constraint {
    Is(b::Type),
    TypeOf(b::ValueIdx),
    Array(b::ValueIdx),
    Ptr(b::ValueIdx),
    ArrayElemPtr(b::ValueIdx),
    ReturnOf(b::ValueIdx),
    ParameterOf(b::ValueIdx, usize),
    IsProperty(b::ValueIdx, String),
    Members(SortedMap<String, b::ValueIdx>),
    HasProperty(String, b::ValueIdx),
    Func(Vec<usize>, usize),
}
impl Constraint {
    pub fn priority(&self) -> ConstraintPriority {
        match self {
            Self::Is(..) => ConstraintPriority::DefinedType,
            Self::TypeOf(..)
            | Self::Array(..)
            | Self::Ptr(..)
            | Self::ArrayElemPtr(..)
            | Self::ReturnOf(..)
            | Self::ParameterOf(..)
            | Self::IsProperty(..) => ConstraintPriority::DerivedDefinedType,
            Self::Members(..) | Self::HasProperty(..) | Self::Func(..) => {
                ConstraintPriority::DerivedInferredType
            }
        }
    }
}
