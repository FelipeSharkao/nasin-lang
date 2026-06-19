use std::borrow::Borrow;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use itertools::chain;

use crate::bytecode as b;

/// Stores the configurations for a compilation
#[derive(Debug, Clone, Default)]
pub struct BuildConfig {
    /// Name of the compilation unit
    pub name: String,
    /// Path where to place the output file
    pub out: PathBuf,
    /// Base directory for the source files
    pub base_dir: PathBuf,
    /// Paths to lookup for libraries
    pub lib_dirs: Vec<PathBuf>,
    /// Omit all messages
    pub silent: bool,
    /// Whether to dump the AST of the source file
    pub dump_ast: DumpFlag,
    /// Whether to dump the bytecode of the source file
    pub dump_bytecode: DumpFlag,
    /// Whether to dump the bytecode of the source file after transformations
    pub dump_transformed_bytecode: DumpFlag,
    /// Whether to dump the parsed bytecode of the source file before type checking
    pub dump_untyped_bytecode: DumpFlag,
    /// Whether to dump the CLIF of the source file, if using Cranelift
    pub dump_clif: bool,
    /// Run the program after compilation
    pub run: bool,
}

impl BuildConfig {
    pub fn base_paths(&self) -> impl IntoIterator<Item = impl AsRef<Path> + '_> + '_ {
        chain!([&self.base_dir], &self.lib_dirs)
    }

    pub fn strip_base_paths<'a>(&'a self, path: &'a Path) -> &'a Path {
        for base_path in self.base_paths() {
            if let Ok(relative_path) = path.strip_prefix(base_path) {
                return relative_path;
            }
        }
        path
    }
}

#[derive(Debug, Clone, Default)]
pub enum DumpFlag {
    #[default]
    None,
    All,
    Modules(HashSet<b::Name>),
}

impl DumpFlag {}

impl From<Option<Option<String>>> for DumpFlag {
    fn from(value: Option<Option<String>>) -> Self {
        match value {
            Some(Some(v)) => {
                let modules = v
                    .trim()
                    .split(',')
                    .map(|module| {
                        let idents = module.split('.').map(|ident| {
                            b::NameIdent::new(ident.to_string(), b::NameIdentKind::Module)
                                .into()
                        });
                        b::Name::new(idents, None)
                    })
                    .collect();
                DumpFlag::Modules(modules)
            }
            Some(None) => DumpFlag::All,
            None => DumpFlag::None,
        }
    }
}

pub trait ShouldDump {
    fn should_dump(&self, module: &b::Name) -> bool;
    fn never_dumps(&self) -> bool;
}

impl ShouldDump for DumpFlag {
    fn should_dump(&self, module: &b::Name) -> bool {
        match self {
            DumpFlag::None => false,
            DumpFlag::All => true,
            DumpFlag::Modules(flag_modules) => {
                for item in flag_modules {
                    if item.starts_with(module) {
                        return true;
                    }
                }
                false
            }
        }
    }

    fn never_dumps(&self) -> bool {
        matches!(self, DumpFlag::None)
    }
}

impl<T: ShouldDump + ?Sized> ShouldDump for &T {
    fn should_dump(&self, module: &b::Name) -> bool {
        (*self).should_dump(module)
    }

    fn never_dumps(&self) -> bool {
        (*self).never_dumps()
    }
}

impl<T: ShouldDump> ShouldDump for [T] {
    fn should_dump(&self, module: &b::Name) -> bool {
        self.iter().any(|flag| flag.borrow().should_dump(module))
    }

    fn never_dumps(&self) -> bool {
        self.iter().all(|flag| flag.borrow().never_dumps())
    }
}
