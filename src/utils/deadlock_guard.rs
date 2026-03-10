use std::backtrace::Backtrace;
use std::sync::Arc;
use std::sync::atomic::{self, AtomicBool};
use std::time::Duration;
use std::{process, thread};

use derive_more::{Deref, DerefMut};

const DEADLOCK_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Deref, DerefMut)]
pub struct DeadlockGuard<T> {
    #[deref(forward)]
    #[deref_mut(forward)]
    lock:      T,
    #[cfg(debug_assertions)]
    is_locked: Arc<AtomicBool>,
}

impl<T> DeadlockGuard<T> {
    pub fn new(lock: T) -> Self {
        let is_locked = Arc::new(AtomicBool::new(true));

        if cfg!(debug_assertions) {
            let backtrace = Backtrace::capture();
            let is_locked = is_locked.clone();
            thread::spawn(move || {
                thread::sleep(DEADLOCK_TIMEOUT);
                if is_locked.load(atomic::Ordering::Relaxed) {
                    panic!("Lock was not released\n{}", backtrace);
                }
            });
        }

        Self { lock, is_locked }
    }
}

#[cfg(debug_assertions)]
impl<T> Drop for DeadlockGuard<T> {
    fn drop(&mut self) {
        self.is_locked.store(false, atomic::Ordering::Relaxed);
    }
}
