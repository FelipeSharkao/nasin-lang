use std::fmt::{self, Display};

pub fn indented<I: IntoIterator>(n: usize, items: I) -> Indented<I::Item, I::IntoIter>
where
    I::Item: Display,
    I::IntoIter: Clone,
{
    Indented {
        indent: n,
        items:  items.into_iter(),
    }
}

pub struct Indented<T: Display, I: Iterator<Item = T> + Clone> {
    indent: usize,
    items:  I,
}

impl<T: Display, I: Iterator<Item = T> + Clone> Display for Indented<T, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let indent = " ".repeat(self.indent);

        for (i, item) in self.items.clone().enumerate() {
            for (j, line) in item.to_string().lines().enumerate() {
                if i > 0 || j > 0 {
                    write!(f, "\n")?;
                }
                write!(f, "{}{}", &indent, line)?;
            }
        }

        Ok(())
    }
}

pub fn join<'a, I: IntoIterator>(sep: &'a str, items: I) -> Join<'a, I::Item, I::IntoIter>
where
    I::Item: Display,
    I::IntoIter: Clone,
{
    Join {
        sep,
        items: items.into_iter(),
    }
}

#[derive(Debug, Clone)]
pub struct Join<'a, T: Display, I: Iterator<Item = T> + Clone> {
    sep:   &'a str,
    items: I,
}

impl<'a, T: Display, I: Iterator<Item = T> + Clone> Display for Join<'a, T, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, item) in self.items.clone().enumerate() {
            if i > 0 {
                write!(f, "{}", self.sep)?;
            }
            write!(f, "{}", item)?;
        }
        Ok(())
    }
}
