use std::path::Path;

/// Returns a relative path if the path is a child of the prefix, or an absolute path
/// otherwise. Assumes that both path are canonicalized and absolute or relative to the
/// same base path.
pub fn relative_path_if_child<P: AsRef<Path>>(path: &Path, prefix: P) -> &Path {
    match path.strip_prefix(prefix) {
        Ok(relative) => relative,
        Err(_) => path,
    }
}
