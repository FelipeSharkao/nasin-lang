use std::fs;
use std::path::Path;

macro_rules! cmd {
    ($cmd:expr $(, $args:expr)* $(; cwd: $cwd:expr)?) => {{
        let mut cmd = ::std::process::Command::new($cmd);
        $(cmd.arg($args);)*
        $(cmd.current_dir($cwd);)?
        match cmd.status() {
            Ok(status) if status.success() => {}
            _ => panic!("failed to run command: {:?}", cmd),
        }
    }};
}

fn main() {
    if cfg!(debug_assertions) || option_env!("GENERATE_GRAMMAR").is_some_and(|x| x == "1")
    {
        if should_compile(
            "tree-sitter-nasin/src/parser.c",
            [
                "tree-sitter-nasin/grammar.js",
                "tree-sitter-nasin/package.json",
            ],
        ) {
            cmd!("bun", "install"; cwd: "tree-sitter-nasin");
            cmd!("bun", "tree-sitter", "generate"; cwd: "tree-sitter-nasin");
            cmd!("bun", "tree-sitter", "build"; cwd: "tree-sitter-nasin");
        }

        println!("cargo:rerun-if-changed=tree-sitter-nasin/grammar.js");
        println!("cargo:rerun-if-changed=tree-sitter-nasin/package.json");
    }

    println!("cargo:rerun-if-changed=tree-sitter-nasin/src/parser.c");
}

fn should_compile(
    target: impl AsRef<Path>,
    deps: impl IntoIterator<Item = impl AsRef<Path>>,
) -> bool {
    let Ok(target_time) = fs::metadata(target).and_then(|m| m.modified()) else {
        return true;
    };
    for dep in deps {
        if fs::metadata(dep)
            .and_then(|m| m.modified())
            .is_ok_and(|dep_time| dep_time > target_time)
        {
            return true;
        }
    }
    false
}
