pub fn on_drop<F: FnMut()>(f: F) -> OnDrop<F> {
    OnDrop { f, active: true }
}

// Returns the value of `f` and calls `on_panic` if the function panics. This doesn't
// catch or stop panics from unwinding.
pub fn with_panic_hook<T>(f: impl FnOnce() -> T, on_panic: impl FnMut()) -> T {
    if cfg!(panic = "abort") {
        return f();
    }
    let mut guard = on_drop(on_panic);
    let ret = f();
    guard.active = false;
    ret
}

pub struct OnDrop<F: FnMut()> {
    f:      F,
    active: bool,
}

impl<F: FnMut()> Drop for OnDrop<F> {
    fn drop(&mut self) {
        if self.active {
            (self.f)();
        }
    }
}
