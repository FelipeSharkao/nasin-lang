use std::path::PathBuf;
use std::sync::Arc;
use std::{fmt, io};

use derive_ctor::ctor;
use derive_more::{Display, From};
use thiserror::Error;

use crate::config::BuildConfig;
use crate::{bytecode as b, sources, utils};

#[derive(Debug, Clone, ctor)]
pub struct Error {
    detail: ErrorDetail,
    loc:    Option<b::Loc>,
}

#[derive(Debug, Clone, Error, ctor)]
pub struct CompilerError {
    source_manager: Option<Arc<sources::SourceManager>>,
    errors: Vec<Error>,
}
impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for err in &self.errors {
            if let (Some(source_manager), Some(loc)) = (&self.source_manager, &err.loc) {
                let idx = loc.source_idx;
                let src = source_manager.source(idx);

                let line = loc.start_line;
                let col = loc.start_col;
                writeln!(
                    f,
                    "{}:{line}:{col} - error: {}",
                    src.path.display(),
                    err.detail
                )?;

                let num = format!("{line}");
                let line_content =
                    src.content().line(line).expect("line should be valid");
                let leading_spaces = line_content
                    .chars()
                    .take_while(|c| c.len_utf8() == 1 && c.is_whitespace())
                    .count();
                writeln!(f, "{num} | {}", &line_content[leading_spaces..])?;
                writeln!(f, "{}^", " ".repeat(num.len() + col - leading_spaces + 2))?;
            } else {
                writeln!(f, "error: {}", &err.detail)?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Display, From)]
pub enum ErrorDetail {
    ReadError(ReadError),
    UnexpectedToken(UnexpectedToken),
    ValueNotFound(ValueNotFound),
    TypeNotFound(TypeNotFound),
    TypeVarNotFound(TypeVarNotFound),
    UnexpectedType(UnexpectedType),
    TypeMisatch(TypeMisatch),
    #[display("Type should be known at this point")]
    TypeNotFinal,
    TypeNotInterface(TypeNotInterface),
    WrongArgumentCount(WrongArgumentCount),
    Todo(Todo),
}

#[derive(Debug, Clone, Display, ctor)]
#[display("Cannot read file `{}`: {kind}", path.display())]
pub struct ReadError {
    pub path: PathBuf,
    pub kind: io::ErrorKind,
}

#[derive(Debug, Clone, Display, ctor)]
#[display("Unexpected token `{}`", token)]
pub struct UnexpectedToken {
    pub token: String,
}

#[derive(Debug, Clone, Display, ctor)]
#[display("Cannot find value `{ident}` on the current scope")]
pub struct ValueNotFound {
    pub ident: String,
}

#[derive(Debug, Clone, Display, ctor)]
#[display("Cannot find type `{ident}` on the current scope")]
pub struct TypeNotFound {
    pub ident: String,
}

#[derive(Debug, Clone, Display, ctor)]
#[display("Cannot find typevar `{ident}` on the current scope")]
pub struct TypeVarNotFound {
    pub ident: String,
}

#[derive(Debug, Clone)]
pub struct UnexpectedType {
    pub expected: Vec<String>,
    pub actual:   String,
}

impl UnexpectedType {
    pub fn new(
        expected: Vec<&b::Type>,
        actual: &b::Type,
        modules: &[b::Module],
        cfg: &BuildConfig,
    ) -> Self {
        let fmt_ty = |ty: &b::TypeBody| {
            let mut s = String::new();
            b::Printer::new(modules, cfg)
                .with_reconstruct(true)
                .write_type_expr(&mut s, ty)
                .unwrap();
            s
        };
        Self {
            expected: expected.iter().map(|t| fmt_ty(&t.body)).collect(),
            actual:   fmt_ty(&actual.body),
        }
    }
}

impl Display for UnexpectedType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.expected.len() == 1 {
            write!(
                f,
                "Expected type {}, but found {} instead",
                &self.expected[0], &self.actual,
            )?;
        } else {
            write!(
                f,
                "Unexpected type {}. Expected one of:\n{}",
                &self.actual,
                utils::indented(2, self.expected.iter().map(|t| format!("- {t}"))),
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TypeMisatch {
    pub types: Vec<String>,
}

impl TypeMisatch {
    pub fn new(types: Vec<&b::Type>, modules: &[b::Module], cfg: &BuildConfig) -> Self {
        Self {
            types: types
                .iter()
                .map(|t| {
                    let mut s = String::new();
                    b::Printer::new(modules, cfg)
                        .with_reconstruct(true)
                        .write_type_expr(&mut s, &t.body)
                        .unwrap();
                    s
                })
                .collect(),
        }
    }
}

impl Display for TypeMisatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "All results of the expression should have the same type\n{}",
            utils::indented(2, self.types.iter().map(|t| format!("- found {t}"))),
        )
    }
}

#[derive(Debug, Clone)]
pub struct TypeNotInterface {
    pub ty: String,
}

impl TypeNotInterface {
    pub fn new(ty: &b::Type, modules: &[b::Module], cfg: &BuildConfig) -> Self {
        let mut s = String::new();
        b::Printer::new(modules, cfg)
            .with_reconstruct(true)
            .write_type_expr(&mut s, &ty.body)
            .unwrap();
        Self { ty: s }
    }
}

impl Display for TypeNotInterface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}` is not an interface type", &self.ty)
    }
}

#[derive(Debug, Clone, Display, ctor)]
#[display(
    "`{name}` requires {expected} {}, but {found} were provided",
    if *expected == 1 { "argument" } else { "arguments" }
)]
pub struct WrongArgumentCount {
    pub name:     String,
    pub expected: usize,
    pub found:    usize,
}

#[derive(Debug, Clone, Display, ctor)]
#[display("Feature is not implemented yet: {feature}")]
pub struct Todo {
    pub feature: String,
}
