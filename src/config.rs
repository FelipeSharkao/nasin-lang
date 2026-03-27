use std::path::{Path, PathBuf};

use itertools::chain;

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
    pub dump_ast: bool,
    /// Whether to dump the bytecode of the source file
    pub dump_bytecode: bool,
    /// Whether to dump the bytecode of the source file after transformations
    pub dump_transformed_bytecode: bool,
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
