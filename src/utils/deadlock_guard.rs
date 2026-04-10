use std::backtrace::Backtrace;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{process, thread};

use derive_more::{Deref, DerefMut};

static CONTROL: Mutex<Option<DeadlockGuardControl>> = Mutex::new(None);

const DEADLOCK_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_LOCKS: usize = 100;

#[derive(Deref, DerefMut)]
pub struct DeadlockGuard<T> {
    #[deref(forward)]
    #[deref_mut(forward)]
    lock:    T,
    #[cfg(debug_assertions)]
    lock_id: usize,
}

impl<T> DeadlockGuard<T> {
    pub fn new(lock: T) -> Self {
        let lock_id = if cfg!(debug_assertions) {
            let item = DeadlockGuardItem {
                backtrace:  Backtrace::capture(),
                created_at: Instant::now(),
            };

            let mut control = CONTROL.lock().unwrap();
            if control.is_none() {
                *control = Some(DeadlockGuardControl {
                    pending_locks: HashMap::new(),
                    next_lock_id:  0,
                });

                thread::spawn(move || {
                    loop {
                        thread::sleep(DEADLOCK_TIMEOUT / 2);
                        let mut control = CONTROL.lock().unwrap();
                        let Some(control) = control.as_mut() else {
                            break;
                        };
                        for item in control.pending_locks.values() {
                            if item.created_at.elapsed() > DEADLOCK_TIMEOUT {
                                eprintln!("Lock was not released");
                                eprintln!("{}", item.backtrace);
                                process::abort();
                            }
                        }
                    }
                });
            }
            let control = control.as_mut().unwrap();

            if control.pending_locks.len() >= MAX_LOCKS {
                eprintln!("Too many locks held, probably a leak");
                for item in control.pending_locks.values().take(5) {
                    eprintln!("{}", item.backtrace);
                }
                eprintln!("Too many locks held, probably a leak");
                process::abort();
            }

            let lock_id = control.next_lock_id;
            control.next_lock_id += 1;
            control.pending_locks.insert(lock_id, item);

            lock_id
        } else {
            0
        };

        Self { lock, lock_id }
    }
}

#[cfg(debug_assertions)]
impl<T> Drop for DeadlockGuard<T> {
    fn drop(&mut self) {
        let mut control = CONTROL.lock().unwrap();
        let Some(control) = control.as_mut() else {
            return;
        };
        control.pending_locks.remove(&self.lock_id);
    }
}

struct DeadlockGuardControl {
    pending_locks: HashMap<usize, DeadlockGuardItem>,
    next_lock_id:  usize,
}

struct DeadlockGuardItem {
    backtrace:  Backtrace,
    created_at: Instant,
}
