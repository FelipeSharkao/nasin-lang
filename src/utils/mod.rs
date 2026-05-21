mod cmd;
mod deadlock_guard;
mod paths;
mod replace_with;
mod scope_stack;
mod sorted_map;
mod str;
mod string_lit;
mod to_radix;
mod traits;
mod write_io;

pub(crate) use self::cmd::*;
pub(crate) use self::deadlock_guard::*;
pub(crate) use self::paths::*;
pub(crate) use self::replace_with::*;
pub(crate) use self::scope_stack::*;
pub(crate) use self::sorted_map::*;
pub(crate) use self::str::*;
pub(crate) use self::string_lit::*;
pub(crate) use self::to_radix::*;
pub(crate) use self::traits::*;
pub(crate) use self::write_io::*;

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
    ($v:expr, $($fmt:tt)*) => {
        $v.unwrap_or_else(|| panic!($($fmt)*))
    };
}
pub(crate) use unwrap;

/// Same as `matches!(expr, pat if guard)`. This only exists because rustfmt breaks with
/// matches with guards
macro_rules! matches_if {
    ($expr:expr, $pat:pat, $guard:expr) => {
        match $expr {
            $pat if $guard => true,
            _ => false,
        }
    };
    ($expr:expr, $pat:pat) => {
        match $expr {
            $pat => true,
            _ => false,
        }
    };
}
pub(crate) use matches_if;
