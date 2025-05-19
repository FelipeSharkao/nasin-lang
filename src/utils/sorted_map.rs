use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

use super::indented;

/// Ordered map that uses  binary search internally. Lookup is O(log(n)), but it
/// guarantees order when iterating. Useful for small maps (<50) that are iterated a lot.
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct SortedMap<K: Ord, V> {
    items: Vec<(K, V)>,
}

impl<K: Ord, V> SortedMap<K, V> {
    pub fn new() -> Self {
        Self { items: vec![] }
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
        }
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        match self.index_for(&key) {
            Ok(idx) => Some(&self.items[idx].1),
            Err(_) => None,
        }
    }
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        match self.index_for(&key) {
            Ok(idx) => Some(&mut self.items[idx].1),
            Err(_) => None,
        }
    }
    pub fn contains<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        self.index_for(&key).is_ok()
    }
    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.items.iter()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut (K, V)> {
        self.items.iter_mut()
    }
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.items.iter().map(|item| &item.0)
    }
    pub fn keys_mut(&mut self) -> impl Iterator<Item = &mut K> {
        self.items.iter_mut().map(|item| &mut item.0)
    }
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.items.iter().map(|item| &item.1)
    }
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.items.iter_mut().map(|item| &mut item.1)
    }
    pub fn insert(&mut self, key: K, value: V) {
        match self.index_for(&key) {
            Ok(idx) => self.items[idx] = (key, value),
            Err(idx) => self.items.insert(idx, (key, value)),
        }
    }
    pub fn remove<Q: ?Sized>(&mut self, key: &Q)
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        if let Ok(idx) = self.index_for(&key) {
            self.items.remove(idx);
        }
    }

    fn index_for<Q: ?Sized>(&self, key: &Q) -> Result<usize, usize>
    where
        K: Borrow<Q>,
        Q: Ord,
    {
        self.items.binary_search_by(|item| item.0.borrow().cmp(key))
    }
}

impl<K: Ord, V> Default for SortedMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, V> From<HashMap<K, V>> for SortedMap<K, V> {
    fn from(value: HashMap<K, V>) -> Self {
        let mut this = Self::with_capacity(value.len());
        this.extend(value);
        this
    }
}

impl<K: Ord, V> From<Vec<(K, V)>> for SortedMap<K, V> {
    fn from(value: Vec<(K, V)>) -> Self {
        let mut this = Self::with_capacity(value.len());
        this.extend(value);
        this
    }
}

impl<K: Ord, V, const N: usize> From<[(K, V); N]> for SortedMap<K, V> {
    fn from(value: [(K, V); N]) -> Self {
        let mut this = Self::with_capacity(value.len());
        this.extend(value);
        this
    }
}

impl<K: Ord, V> From<HashSet<(K, V)>> for SortedMap<K, V> {
    fn from(value: HashSet<(K, V)>) -> Self {
        let mut this = Self::with_capacity(value.len());
        this.extend(value);
        this
    }
}

impl<K: Ord, V, Q: ?Sized> Index<&Q> for SortedMap<K, V>
where
    K: Borrow<Q>,
    Q: Ord,
{
    type Output = V;
    fn index(&self, index: &Q) -> &Self::Output {
        self.get(index).expect("no entry found for key")
    }
}

impl<K: Ord, V, Q: ?Sized> IndexMut<&Q> for SortedMap<K, V>
where
    K: Borrow<Q>,
    Q: Ord,
{
    fn index_mut(&mut self, index: &Q) -> &mut Self::Output {
        self.get_mut(index).expect("no entry found for key")
    }
}

impl<K: Ord, V> IntoIterator for SortedMap<K, V> {
    type Item = (K, V);
    type IntoIter = <Vec<(K, V)> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

impl<'a, K: Ord, V> IntoIterator for &'a SortedMap<K, V> {
    type Item = &'a (K, V);
    type IntoIter = <&'a Vec<(K, V)> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        (&self.items).into_iter()
    }
}

impl<'a, K: Ord, V> IntoIterator for &'a mut SortedMap<K, V> {
    type Item = &'a mut (K, V);
    type IntoIter = <&'a mut Vec<(K, V)> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        (&mut self.items).into_iter()
    }
}

impl<K: Ord, V> Extend<(K, V)> for SortedMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for SortedMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut this = Self::new();
        this.extend(iter);
        this
    }
}

impl<K: Ord + Debug, V: Debug> Debug for SortedMap<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SortedMap {{")?;
        if self.items.len() == 1 {
            let (key, value) = &self.items[0];
            write!(f, "{key:?}: {value:?}")?;
        } else if self.items.len() > 1 {
            writeln!(
                f,
                "\n{}",
                indented(2, self.iter().map(|(k, v)| format!("{k:?}: {v:?}")))
            )?;
        }
        write!(f, "}}")?;
        Ok(())
    }
}
