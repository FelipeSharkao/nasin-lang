macro_rules! cmd {
    ($prog:expr $(, $args:expr )* $(,)?) => {{
        let mut cmd = ::std::process::Command::new($prog);
        $(
            cmd.arg($args);
        )*
        cmd
    }};
    ($cmd:expr; $( $args:expr ),+ $(,)?) => {{
        let mut cmd: ::std::process::Command = $cmd;
        $(
            cmd.arg($args);
        )*
        cmd
    }};
}
pub(crate) use cmd;
