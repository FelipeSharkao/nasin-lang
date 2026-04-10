use std::cmp::min;
use std::fmt::Debug;

use derive_ctor::ctor;
use derive_more::{Deref, DerefMut, IntoIterator};

use crate::bytecode as b;

#[derive(Debug, Clone, PartialEq, Eq, IntoIterator)]
pub struct ScopeStack<T: ScopePayload> {
    #[into_iterator(owned, ref, ref_mut)]
    scopes: Vec<Scope<T>>,
}

impl<T: ScopePayload> ScopeStack<T> {
    pub fn new(initial_payload: T, block_idx: b::BlockIdx) -> Self {
        Self {
            scopes: vec![Scope::new(initial_payload, block_idx)],
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

    pub fn find_last_index(&self, mut f: impl FnMut(&Scope<T>) -> bool) -> Option<usize> {
        (0..self.scopes.len()).rev().find(|i| f(&self.scopes[*i]))
    }

    pub fn find_last(&self, f: impl FnMut(&Scope<T>) -> bool) -> Option<&Scope<T>> {
        self.find_last_index(f).map(|i| &self.scopes[i])
    }

    pub fn find_last_mut(
        &mut self,
        f: impl FnMut(&Scope<T>) -> bool,
    ) -> Option<&mut Scope<T>> {
        self.find_last_index(f).map(|i| &mut self.scopes[i])
    }

    pub fn begin(&mut self, payload: T, block_idx: b::BlockIdx) -> &mut Scope<T> {
        self.scopes.push(Scope::new(payload, block_idx));
        self.scopes.last_mut().unwrap()
    }

    pub fn begin_cloned(&mut self, block_idx: b::BlockIdx) -> &mut Scope<T>
    where
        T: Clone,
    {
        self.begin(self.last().payload.clone(), block_idx)
    }

    pub fn branch(&mut self, block_idx: b::BlockIdx) -> (&mut Scope<T>, T::Result) {
        assert!(
            self.scopes.len() > 1,
            "should not brach the first scope of function"
        );

        let mut scope = self.scopes.pop().unwrap();
        scope.block_idx = block_idx;
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
            if let Some(last) = self.scopes.last_mut() {
                last.mark_as_never();
            }
        }

        let mut prev = self.scopes.pop();

        scope.payload.end(prev.as_mut().map(|x| &mut x.payload));
        let payload_result = scope.payload.reset(prev.as_ref().map(|x| &x.payload));

        self.scopes.extend(prev);

        (scope, payload_result)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deref, DerefMut, ctor)]
pub struct Scope<T> {
    #[deref]
    #[deref_mut]
    pub payload: T,
    pub block_idx: b::BlockIdx,
    #[ctor(expr(1))]
    branches: usize,
    #[ctor(expr(0))]
    never_branches: usize,
    #[ctor(expr(false))]
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

    fn end(&mut self, prev: Option<&mut Self>) {
        let _ = prev;
    }
}
