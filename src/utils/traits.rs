use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::{self, Display, Write};

use derive_ctor::ctor;
use tree_sitter as ts;

pub trait TreeSitterUtils<'t> {
    fn of_kind(self, kind: &str) -> Self;
    fn get_text<'s>(&self, source: &'s str) -> &'s str;
    fn iter_children(&self) -> impl Iterator<Item = ts::Node<'t>>;
    fn iter_field(&self, field: &str) -> impl Iterator<Item = ts::Node<'t>>;
    fn iter_errors(&self) -> impl Iterator<Item = ts::Node<'t>>;
    fn field(&self, field: &str) -> Option<ts::Node<'t>>;
    fn required_field(&self, field: &str) -> ts::Node<'t>;
    fn display<'s>(&self, source: &'s str) -> impl Display;
}

impl<'t> TreeSitterUtils<'t> for ts::Node<'t> {
    fn of_kind(self, kind: &str) -> Self {
        assert!(self.is_named());
        assert_eq!(self.kind(), kind);
        self
    }

    fn get_text<'s>(&self, source: &'s str) -> &'s str {
        &source[self.start_byte()..self.end_byte()]
    }

    fn iter_children(&self) -> impl Iterator<Item = ts::Node<'t>> {
        TreeSitterChildren::new(self).map(|x| x.node)
    }

    fn iter_field(&self, field: &str) -> impl Iterator<Item = ts::Node<'t>> {
        let field = field.to_string();
        TreeSitterChildren::new(self)
            .filter(move |x| x.field.is_some_and(|f| f == field))
            .map(|x| x.node)
    }

    fn iter_errors(&self) -> impl Iterator<Item = ts::Node<'t>> {
        TreeSitterErrors::new(self)
    }

    fn field(&self, field: &str) -> Option<ts::Node<'t>> {
        self.iter_field(field).next()
    }

    fn required_field(&self, field: &str) -> ts::Node<'t> {
        self.field(field)
            .expect(&format!("Field {} is missing", field))
    }

    fn display<'s>(&self, source: &'s str) -> impl Display {
        TreeSitterDisplay::new(*self, source)
    }
}

struct TreeSitterChildren<'t> {
    cursor:   ts::TreeCursor<'t>,
    finished: bool,
}

struct TreeSitterChild<'t> {
    node:  ts::Node<'t>,
    field: Option<&'static str>,
}

impl<'t> TreeSitterChildren<'t> {
    fn new(node: &ts::Node<'t>) -> Self {
        let mut cursor = node.walk();
        let has_children = cursor.goto_first_child();

        Self {
            cursor,
            finished: !has_children,
        }
    }
}

impl<'t> Iterator for TreeSitterChildren<'t> {
    type Item = TreeSitterChild<'t>;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.finished && !self.cursor.node().is_named() {
            self.finished = !self.cursor.goto_next_sibling();
        }

        if self.finished {
            return None;
        }

        let node = self.cursor.node();
        let field = self.cursor.field_name();

        self.finished = !self.cursor.goto_next_sibling();

        Some(TreeSitterChild { node, field })
    }
}

struct TreeSitterErrors<'t> {
    cursor:   ts::TreeCursor<'t>,
    root:     ts::Node<'t>,
    finished: bool,
}

impl<'t> TreeSitterErrors<'t> {
    fn new(node: &ts::Node<'t>) -> Self {
        let mut cursor = node.walk();
        let has_children = cursor.goto_first_child();

        Self {
            cursor,
            root: *node,
            finished: !has_children,
        }
    }

    fn goto_next_sibling_or_parent(&mut self) {
        while !self.finished && !self.cursor.goto_next_sibling() {
            self.cursor.goto_parent();
            self.finished = self.cursor.node() == self.root;
        }
    }
}

impl<'t> Iterator for TreeSitterErrors<'t> {
    type Item = ts::Node<'t>;

    fn next(&mut self) -> Option<Self::Item> {
        while !self.finished
            && !self.cursor.node().is_error()
            && !self.cursor.node().is_missing()
        {
            if self.cursor.node().has_error() {
                self.cursor.goto_first_child();
                continue;
            }
            self.goto_next_sibling_or_parent();
        }

        if self.finished {
            return None;
        }

        let node = self.cursor.node();

        self.goto_next_sibling_or_parent();

        Some(node)
    }
}

pub trait IntoItem<Q> {
    type Item;
    fn into_item(self, item: Q) -> Option<Self::Item>;
}
impl<T, I: IntoIterator<Item = T>> IntoItem<usize> for I {
    type Item = T;
    fn into_item(self, n: usize) -> Option<Self::Item> {
        self.into_iter().nth(n)
    }
}

const S: &str = "";

#[derive(ctor)]
struct TreeSitterDisplay<'t, 's> {
    node: ts::Node<'t>,
    source: &'s str,
    #[ctor(default)]
    temp_strings: RefCell<Vec<String>>,
}

impl<'t, 's> TreeSitterDisplay<'t, 's> {
    fn write_node(
        &self,
        f: &mut impl Write,
        node: ts::Node<'_>,
        indent: usize,
    ) -> fmt::Result {
        let kind = node.kind();
        write!(f, "({kind}")?;

        let mut children_str = self.temp_string();
        let mut children = TreeSitterChildren::new(&node);
        while let Some(child) = children.next() {
            write!(children_str, "\n{S:indent$}  ")?;
            if let Some(field) = child.field {
                write!(children_str, "{field}: ")?;
            }
            self.write_node(&mut children_str, child.node, indent + 2)?;
        }

        if children_str.is_empty() {
            let text = node.get_text(self.source);
            write!(f, " {text:?}")?;
        }

        let start = node.start_position();
        let end = node.end_position();
        write!(
            f,
            " [{}, {}] - [{}, {}]",
            start.row, start.column, end.row, end.column,
        )?;

        if !children_str.is_empty() {
            write!(f, "{children_str}")?;
        }

        self.drop_temp_string(children_str);

        write!(f, ")")
    }

    fn temp_string(&self) -> String {
        self.temp_strings.borrow_mut().pop().unwrap_or_default()
    }

    fn drop_temp_string(&self, mut s: String) {
        s.clear();
        self.temp_strings.borrow_mut().push(s);
    }
}

impl<'t, 's> Display for TreeSitterDisplay<'t, 's> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_node(f, self.node, 0)
    }
}
