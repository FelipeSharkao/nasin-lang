use std::collections::HashSet;

use derive_new::new;

use super::{Loc, Type};

#[derive(Debug, Clone, new)]
pub struct Value {
    pub ty: Type,
    pub loc: Option<Loc>,
    /// Indicates that a value is not in use anymore and could be stripped during
    /// optimizations.
    #[new(value = "false")]
    pub dead: bool,
    /// Indicates that a value was merged with another. Every processing and modification
    /// should be applied to that one instead. If present, this value may be removed and
    /// its uses replaced with the target value.
    #[new(default)]
    pub redirects_to: Option<ValueIdx>,
    /// Indicates that a value type was merged with another. Every processing and
    /// modification on the type should be applied to those ones instead. If it contains
    /// more than one value, it describes that it is dynamically one of them, and it's
    /// type should be a union the types of the referenced values.
    #[new(default)]
    pub same_type_of: HashSet<ValueIdx>,
}

pub type ValueIdx = usize;
