use std::borrow::Cow;

use tree_sitter as ts;

pub trait TreeSitterUtils<'t> {
    fn of_kind(self, kind: &str) -> Self;
    fn get_text<'s>(&self, source: &'s str) -> &'s str;
    fn iter_children(&self) -> impl Iterator<Item = ts::Node<'t>>;
    fn iter_field(&self, field: &str) -> impl Iterator<Item = ts::Node<'t>>;
    fn field(&self, field: &str) -> Option<ts::Node<'t>>;
    fn required_field(&self, field: &str) -> ts::Node<'t>;
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
        TreeSitterChildren::new(self, None)
    }

    fn iter_field(&self, field: &str) -> impl Iterator<Item = ts::Node<'t>> {
        TreeSitterChildren::new(self, Some(field.to_string()))
    }

    fn field(&self, field: &str) -> Option<ts::Node<'t>> {
        self.iter_field(field).next()
    }

    fn required_field(&self, field: &str) -> ts::Node<'t> {
        self.field(field)
            .expect(&format!("Field {} is missing", field))
    }
}

struct TreeSitterChildren<'t> {
    field:    Option<String>,
    cursor:   ts::TreeCursor<'t>,
    finished: bool,
}
impl<'t> TreeSitterChildren<'t> {
    fn new(node: &ts::Node<'t>, field: Option<String>) -> Self {
        let mut cursor = node.walk();
        let has_children = cursor.goto_first_child();

        TreeSitterChildren {
            field,
            cursor,
            finished: !has_children,
        }
    }
}
impl<'t> Iterator for TreeSitterChildren<'t> {
    type Item = ts::Node<'t>;

    fn next(&mut self) -> Option<Self::Item> {
        // Skip to a valid node
        while !self.finished {
            match (self.cursor.node().is_named(), self.field.as_ref()) {
                (true, Some(field))
                    if self.cursor.field_name().is_some_and(|f| f == field) =>
                {
                    break
                }
                (true, None) => break,
                _ => {
                    self.finished = !self.cursor.goto_next_sibling();
                }
            }
        }

        if self.finished {
            return None;
        }

        let node = self.cursor.node();

        self.finished = !self.cursor.goto_next_sibling();

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
