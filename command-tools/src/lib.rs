use std::ffi::{OsStr, OsString};
use std::fmt::Display;
use std::io;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{self, Command, ExitStatus};

use colored::Colorize;

#[macro_export]
macro_rules! cmd {
    ($prog:expr $(, $arg:expr)* $(,)?) => {{
        #[allow(unused_mut)]
        let mut cmd = ::std::process::Command::new($prog);
        $(::command_tools::write_command_args(&mut cmd, &$arg);)*
        cmd
    }};
}

pub trait CommandTools {
    /// Executes the command as a child process, waiting for it to finish and returning a
    /// result. By default, stdin, stdout and stderr are inherited from the parent.
    ///
    /// # Errors
    ///
    /// This function will return an error if the command could not be executed or if
    /// exits with a non-zero status.
    fn exec(&mut self) -> Result<(), CommandError>;
    /// Echos the command to stderr then executes it as a child process, waiting for it to
    /// finish. This is mainly intended for scripts. By default, stdin, stdout and stderr
    /// are inherited from the parent.
    ///
    /// # Panics
    ///
    /// This function will panic if the command could not be executed or if exits with a
    /// non-zero status.
    fn run(&mut self) -> &mut Self;
}

impl CommandTools for Command {
    fn exec(&mut self) -> Result<(), CommandError> {
        match self.status() {
            Ok(status) if status.success() => Ok(()),
            Ok(status) => Err(status.into()),
            Err(err) => Err(CommandError::Read(err.kind())),
        }
    }

    fn run(&mut self) -> &mut Self {
        eprintln!("{}", format!("$ {self:?}").dimmed());
        let name = self.get_program().to_string_lossy().into_owned();
        match self.exec() {
            Ok(()) => self,
            Err(err) => panic!("{name}: {err}"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum CommandError {
    Read(io::ErrorKind),
    Code(i32),
    Signal(i32),
    Segfault,
}

impl Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommandError::Read(kind) => write!(f, "{kind}")?,
            CommandError::Code(code) => write!(f, "exited with {code}")?,
            CommandError::Signal(signal) => write!(f, "terminated by signal {signal}")?,
            CommandError::Segfault => write!(f, "Segmentation fault")?,
        }
        Ok(())
    }
}

impl CommandError {
    pub fn exit_process(self) {
        match self {
            CommandError::Read(_) => process::exit(1),
            CommandError::Code(code) => process::exit(code),
            CommandError::Signal(signal) => process::exit(128 + signal),
            CommandError::Segfault => process::exit(139),
        }
    }
}

impl From<ExitStatus> for CommandError {
    fn from(status: ExitStatus) -> Self {
        if let Some(signal) = status.signal() {
            if signal == 11 {
                CommandError::Segfault
            } else {
                CommandError::Signal(signal)
            }
        } else {
            CommandError::Code(status.code().unwrap_or(-1))
        }
    }
}

pub trait WriteCommandArgs {
    fn write_command_args(&self, command: &mut Command);
}

pub fn write_command_args(command: &mut Command, args: &impl WriteCommandArgs) {
    args.write_command_args(command);
}

impl<T: WriteCommandArgs> WriteCommandArgs for &T {
    fn write_command_args(&self, command: &mut Command) {
        (*self).write_command_args(command);
    }
}

macro_rules! impl_write_command_arg {
    ($ty:ty) => {
        impl WriteCommandArgs for $ty {
            fn write_command_args(&self, command: &mut Command) {
                command.arg(self);
            }
        }
    };
}

impl_write_command_arg!(&str);
impl_write_command_arg!(String);
impl_write_command_arg!(&OsStr);
impl_write_command_arg!(OsString);
impl_write_command_arg!(&Path);
impl_write_command_arg!(PathBuf);

macro_rules! impl_write_command_args {
    ($ty:ty $(; $($arg:tt)*)?) => {
        impl $(<$($arg)*>)? WriteCommandArgs for $ty {
            fn write_command_args(&self, command: &mut Command) {
                for arg in self.into_iter() {
                    arg.write_command_args(command);
                }
            }
        }
    };
}

impl_write_command_args!(Option<T>; T: WriteCommandArgs);
impl_write_command_args!(Vec<T>; T: WriteCommandArgs);
impl_write_command_args!(&[T]; T: WriteCommandArgs);
impl_write_command_args!([T; N]; T: WriteCommandArgs, const N: usize);
