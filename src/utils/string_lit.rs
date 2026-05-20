use std::fmt::{self, Display};

use derive_ctor::ctor;

pub fn encode_string_lit(s: &str) -> StringLitEncoder<'_> {
    StringLitEncoder { string: s }
}

pub fn decode_string_lit(s: &str) -> StringLitDecoder<'_> {
    StringLitDecoder { lit: s }
}

#[derive(Clone, Copy, ctor)]
pub struct StringLitEncoder<'a> {
    string: &'a str,
}

impl<'a> Display for StringLitEncoder<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"")?;

        for c in self.string.chars() {
            match c {
                '\n' => write!(f, "\\n")?,
                '\r' => write!(f, "\\r")?,
                '\t' => write!(f, "\\t")?,
                '\\' => write!(f, "\\\\")?,
                '\0' => write!(f, "\\0")?,
                '"' => write!(f, "\\\"")?,
                c => write!(f, "{c}")?,
            }
        }

        write!(f, "\"")?;

        Ok(())
    }
}

#[derive(Clone, Copy, ctor)]
pub struct StringLitDecoder<'a> {
    lit: &'a str,
}

impl<'a> Display for StringLitDecoder<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut chars = self.lit.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                let c = chars.next().unwrap();
                match c {
                    'n' => write!(f, "\n")?,
                    'r' => write!(f, "\r")?,
                    't' => write!(f, "\t")?,
                    '\\' => write!(f, "\\")?,
                    '0' => write!(f, "\0")?,
                    _ => panic!("Unknown escape sequence: \\{c}"),
                }
            } else {
                write!(f, "{c}")?;
            }
        }

        Ok(())
    }
}
