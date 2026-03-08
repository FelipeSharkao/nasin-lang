#![allow(unused_imports)]

mod cmd;
mod enumerate;
mod paths;
mod replace_with;
mod scope_stack;
mod sorted_map;
mod str;
mod string_lit;
mod to_radix;
mod traits;

pub use self::cmd::*;
pub use self::enumerate::*;
pub use self::paths::*;
pub use self::replace_with::*;
pub use self::scope_stack::*;
pub use self::sorted_map::*;
pub use self::str::*;
pub use self::string_lit::*;
pub use self::to_radix::*;
pub use self::traits::*;

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
