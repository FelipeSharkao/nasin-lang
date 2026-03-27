use std::{fmt, io};

pub struct WriteIO<W: io::Write>(W);

impl<W: io::Write> fmt::Write for WriteIO<W> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0.write_all(s.as_bytes()).map_err(|_| fmt::Error)?;
        Ok(())
    }
}

impl WriteIO<io::Stdout> {
    pub fn stdout() -> WriteIO<io::Stdout> {
        WriteIO(io::stdout())
    }
}
