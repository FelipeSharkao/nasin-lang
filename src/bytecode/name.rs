use std::borrow::Cow;
use std::fmt;
use std::path::{Component, Path};

use derive_ctor::ctor;
use derive_more::{Debug, Display, From};
use itertools::{Itertools, izip};

use super::module::*;
use super::ty::*;
use crate::config::BuildConfig;
use crate::utils;

#[derive(Debug, Clone, PartialEq, Eq, Hash, ctor)]
pub struct Name {
    #[ctor(iter(NameNode))]
    pub nodes: Vec<NameNode>,
    pub loc:   Option<Loc>,
}

impl Name {
    /// Create a new name from a identifier
    pub fn from_ident(
        ident: impl Into<String>,
        kind: NameIdentKind,
        loc: Option<Loc>,
    ) -> Self {
        Self {
            nodes: vec![NameIdent::new(ident.into(), kind).into()],
            loc,
        }
    }

    /// Create a new name by appending a new identifier to the end
    pub fn with(
        &self,
        ident: impl Into<String>,
        kind: NameIdentKind,
        loc: Option<Loc>,
    ) -> Self {
        let mut nodes = self.nodes.clone();
        nodes.push(NameIdent::new(ident.into(), kind).into());
        Self {
            nodes,
            loc: loc.or(self.loc),
        }
    }

    /// Resolves the name to a path. Uses the base paths to resolve relative paths.
    /// Assumes that all paths are absolute and canonicalized.
    pub fn from_path(path: &Path, cfg: &BuildConfig) -> Self {
        let path = cfg.strip_base_paths(path);
        let nodes = path
            .parent()
            .iter()
            .flat_map(|p| p.components())
            .filter(|c| *c != Component::CurDir)
            .map(|c| c.as_os_str())
            .chain(path.file_stem())
            .map(|c| {
                NameIdent::new(c.to_string_lossy().to_string(), NameIdentKind::Module)
                    .into()
            })
            .collect_vec();
        Self { nodes, loc: None }
    }

    /// Creates a new name by appending a new type parameter list to the end. If the last
    /// node is a type parameter, it will be replaced instead of creating a new type
    /// parameter list.
    pub fn with_type_params(
        &self,
        tys: impl IntoIterator<Item = Type>,
        loc: Option<Loc>,
    ) -> Self {
        let mut nodes = self.nodes.clone();
        if let Some(NameNode::TypeParams(params)) = nodes.last_mut() {
            params.params.splice(0..params.params.len(), tys);
        } else {
            nodes.push(NameTypeParams::new(tys.into_iter().collect()).into());
        }
        Self {
            nodes,
            loc: loc.or(self.loc),
        }
    }

    /// Create a new name by appending a new type parameter list to the end. If the last
    /// node is a type parameter, it will be inserted at the end of the list instead of
    /// creating a new type parameter list.
    pub fn with_new_type_params(
        &self,
        tys: impl IntoIterator<Item = Type>,
        loc: Option<Loc>,
    ) -> Self {
        let mut nodes = self.nodes.clone();
        if let Some(NameNode::TypeParams(params)) = nodes.last_mut() {
            params.params.extend(tys);
        } else {
            nodes.push(NameTypeParams::new(tys.into_iter().collect()).into());
        }
        Self {
            nodes,
            loc: loc.or(self.loc),
        }
    }

    /// Get the last identifier of the name
    pub fn last_ident(&self) -> &str {
        for node in self.nodes.iter().rev() {
            if let NameNode::Ident(ident) = node {
                return &ident.ident;
            }
        }
        panic!("Name is empty")
    }

    pub fn strip_prefix(&self, prefix: &Self) -> Cow<'_, Self> {
        if self.nodes.len() < prefix.nodes.len() {
            return Cow::Borrowed(self);
        }
        for (a, b) in izip!(&self.nodes, &prefix.nodes) {
            if a != b {
                return Cow::Borrowed(self);
            }
        }
        Cow::Owned(Self {
            nodes: self.nodes[prefix.nodes.len()..].to_vec(),
            ..self.clone()
        })
    }
}

impl Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.nodes.iter().peekable();
        while let Some(node) = iter.next() {
            write!(f, "{node}")?;
            if let Some(NameNode::Ident(_)) = iter.peek() {
                write!(f, ".")?;
            }
        }
        Ok(())
    }
}

#[derive(Display, Debug, Clone, PartialEq, Eq, Hash, From)]
pub enum NameNode {
    Ident(NameIdent),
    TypeParams(NameTypeParams),
}

#[derive(Display, Debug, Clone, PartialEq, Eq, Hash, ctor)]
#[display("{ident}")]
pub struct NameIdent {
    pub ident: String,
    pub kind:  NameIdentKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NameIdentKind {
    Module,
    Type,
    Value,
    Func,
}

#[derive(Display, Debug, Clone, PartialEq, Eq, Hash, ctor)]
#[display("<{}>", utils::join(", ", params))]
pub struct NameTypeParams {
    pub params: Vec<Type>,
}
