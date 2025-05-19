use derive_new::new;

use crate::bytecode as b;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ValueRefBody {
    Func(usize, usize),
    Global(usize, usize),
    Value(b::ValueIdx),
    Bool(bool),
    Number(String),
    Never,
    CompileError,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, new)]
pub struct ValueRef {
    pub body: ValueRefBody,
    #[new(default)]
    pub ty:   Option<b::Type>,
    pub loc:  b::Loc,
}
impl ValueRef {
    pub fn is_never(&self) -> bool {
        matches!(&self.body, ValueRefBody::Never)
    }
    pub fn with_loc(&self, loc: b::Loc) -> Self {
        let mut new = self.clone();
        new.loc = loc;
        new
    }
}
