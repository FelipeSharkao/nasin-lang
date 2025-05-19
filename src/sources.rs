use std::fs::File;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};

use derivative::Derivative;
use derive_new::new;
use itertools::Itertools;
use lazy_init::LazyTransform;

use crate::bytecode as b;

#[derive(Default, Debug, new)]
pub struct SourceManager {
    #[new(default)]
    pub sources: Vec<Source>,
}
impl SourceManager {
    pub fn source(&self, idx: usize) -> &Source {
        &self.sources.get(idx).expect("source should exist")
    }
    pub fn find_source(&self, path: &Path) -> Option<&Source> {
        self.sources.iter().find(|item| item.path == path)
    }
    pub fn open(&mut self, path: PathBuf) -> io::Result<usize> {
        self.sources.push(Source::open(path)?);
        Ok(self.sources.len() - 1)
    }
    pub fn preload(&mut self, path: PathBuf) -> io::Result<usize> {
        let idx = self.open(path)?;
        self.source(idx).content();
        Ok(idx)
    }
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct Source {
    pub path: PathBuf,
    #[derivative(Debug = "ignore")]
    content:  LazyTransform<File, SourceContent>,
}
impl Source {
    pub fn open(path: PathBuf) -> io::Result<Self> {
        Ok(Self {
            content: LazyTransform::new(File::open(&path)?),
            path,
        })
    }
    pub fn content(&self) -> &SourceContent {
        self.content.get_or_create(|mut file| {
            let mut buf = "".to_string();
            file.read_to_string(&mut buf)
                .expect("file should be readable");
            SourceContent::new(buf)
        })
    }
}
impl From<&Source> for b::Source {
    fn from(value: &Source) -> Self {
        b::Source::new(value.path.clone())
    }
}

pub struct SourceContent {
    pub text: String,
    /// Indexes of each line after the first (which is implied to be at index 0)
    lines:    Vec<usize>,
}
impl SourceContent {
    pub fn new(text: String) -> Self {
        let lines = text.match_indices('\n').map(|(i, _)| i + 1).collect_vec();

        Self { text, lines }
    }
    pub fn line(&self, n: usize) -> Option<&str> {
        if n > self.lines.len() + 1 {
            return None;
        }

        // Line 1 is not in the array
        // Line 2 is at index 0
        // Line 3 is at index 1
        // ...
        let start = if n > 1 { self.lines[n - 2] } else { 0 };
        let end = if n < self.lines.len() + 1 {
            self.lines[n - 1] - 1
        } else {
            self.text.len()
        };

        Some(&self.text[start..end])
    }
}
