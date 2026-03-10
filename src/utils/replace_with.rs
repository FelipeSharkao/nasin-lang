use std::ptr;

/// Temporarily takes ownership of a value at a mutable location, and replace it with a
/// new value based on the old one.
pub fn replace_with<'a, T, U, R: ReplaceResult<T, U>>(
    dest: &mut T,
    f: impl (FnOnce(T) -> R) + 'a,
) -> U {
    unsafe {
        let old = ptr::read(dest);
        let (ctor, ret) = f(old).get_replace_result();
        ptr::write(dest, ctor);
        ret
    }
}

pub trait ReplaceResult<T, U> {
    fn get_replace_result(self) -> (T, U);
}

impl<T> ReplaceResult<T, ()> for T {
    fn get_replace_result(self) -> (T, ()) {
        (self, ())
    }
}
impl<T, R1> ReplaceResult<T, R1> for (T, R1) {
    fn get_replace_result(self) -> (T, R1) {
        (self.0, self.1)
    }
}
