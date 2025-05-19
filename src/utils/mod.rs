#![allow(unused_imports)]

mod enumerate;
mod idented;
mod replace_with;
mod scope_stack;
mod sorted_map;
mod string_lit;
mod traits;

pub use enumerate::*;
pub use idented::*;
pub use replace_with::*;
pub use scope_stack::*;
pub use sorted_map::*;
pub use string_lit::*;
pub use traits::*;

macro_rules! unordered {
    ($a:pat, $b:pat $(,)?) => {
        ($a, $b) | ($b, $a)
    };
}
pub(crate) use unordered;

/// Unwrap a value or panics with a formatted message if not possible. Requires the value
/// to have the methods `.unwrap()` and `.unwrap_or_else(f)`
macro_rules! unwrap {
    ($v:expr) => {
        $v.unwrap()
    };
    ($v:expr, $msg:literal $(, $fmt:tt)* $(,)?) => {
        $v.unwrap_or_else(|| panic!($msg, $($fmt),*))
    };
}
pub(crate) use unwrap;
