use std::cmp::min;
use std::fmt::Debug;

use derive_more::{Deref, DerefMut, IntoIterator};
use derive_new::new;

#[derive(Debug, Clone, PartialEq, Eq, IntoIterator)]
pub struct ScopeStack<T: ScopePayload> {
    #[into_iterator(owned, ref, ref_mut)]
    scopes: Vec<Scope<T>>,
}

impl<T: ScopePayload> ScopeStack<T> {
    pub fn new(initial_payload: T) -> Self {
        Self {
            scopes: vec![Scope::new(initial_payload)],
        }
    }

    pub fn empty() -> Self {
        Self { scopes: vec![] }
    }

    pub fn len(&self) -> usize {
        self.scopes.len()
    }

    pub fn get(&self, idx: usize) -> Option<&Scope<T>> {
        self.scopes.get(idx)
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut Scope<T>> {
        self.scopes.get_mut(idx)
    }

    pub fn first(&self) -> &Scope<T> {
        self.scopes.first().unwrap()
    }

    pub fn first_mut(&mut self) -> &mut Scope<T> {
        self.scopes.first_mut().unwrap()
    }

    pub fn last(&self) -> &Scope<T> {
        self.scopes.last().unwrap()
    }

    pub fn last_mut(&mut self) -> &mut Scope<T> {
        self.scopes.last_mut().unwrap()
    }

    pub fn last_loop(&self) -> Option<&Scope<T>> {
        self.scopes.iter().rev().find(|scope| scope.is_loop)
    }

    pub fn last_loop_mut(&mut self) -> Option<&mut Scope<T>> {
        self.scopes.iter_mut().rev().find(|scope| scope.is_loop)
    }

    pub fn begin(&mut self, payload: T) -> &mut Scope<T> {
        self.scopes.push(Scope::new(payload));
        self.scopes.last_mut().unwrap()
    }

    pub fn begin_cloned(&mut self) -> &mut Scope<T>
    where
        T: Clone,
    {
        self.begin(self.last().payload.clone())
    }

    pub fn branch(&mut self) -> (&mut Scope<T>, T::Result) {
        assert!(
            self.scopes.len() > 1,
            "should not brach the first scope of function"
        );

        let mut scope = self.scopes.pop().unwrap();
        scope.branches += 1;
        scope.is_never = false;

        let prev = self.scopes.pop();
        let prev_payload = prev.as_ref().map(|x| &x.payload);
        scope.payload.branch(prev_payload);
        let payload_result = scope.payload.reset(prev_payload);

        self.scopes.extend(prev);
        self.scopes.push(scope);

        let scope = self.scopes.last_mut().unwrap();

        (scope, payload_result)
    }

    pub fn end(&mut self) -> (Scope<T>, T::Result) {
        assert!(
            self.scopes.len() > 0,
            "should not try to end with no scopes"
        );
        let mut scope = self.scopes.pop().unwrap();

        scope.is_never = scope.never_branches == scope.branches;
        if scope.is_never {
            self.last_mut().mark_as_never();
        }

        let prev = self.scopes.pop();
        let prev_payload = prev.as_ref().map(|x| &x.payload);
        let payload_result = scope.payload.reset(prev_payload);

        self.scopes.extend(prev);

        (scope, payload_result)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deref, DerefMut, new)]
pub struct Scope<T> {
    #[deref]
    #[deref_mut]
    pub payload: T,
    #[new(value = "false")]
    pub is_loop: bool,
    #[new(value = "0")]
    pub loop_arity: usize,
    #[new(value = "1")]
    branches: usize,
    #[new(value = "0")]
    never_branches: usize,
    #[new(value = "false")]
    is_never: bool,
}

impl<T> Scope<T> {
    pub fn is_never(&self) -> bool {
        self.is_never
    }

    pub fn mark_as_never(&mut self) {
        self.is_never = true;
        self.never_branches += 1;
    }
}

pub trait ScopePayload: Debug {
    type Result;
    fn reset(&mut self, prev: Option<&Self>) -> Self::Result;
    fn branch(&mut self, prev: Option<&Self>) {
        let _ = prev;
    }
}

pub trait SimpleScopePayload: Debug {
    fn branch(&mut self, _prev: Option<&Self>) {}
}
impl<T> ScopePayload for T
where
    T: SimpleScopePayload,
{
    type Result = ();
    fn reset(&mut self, _: Option<&Self>) -> Self::Result {
        ()
    }
    fn branch(&mut self, prev: Option<&Self>) {
        SimpleScopePayload::branch(self, prev);
    }
}
