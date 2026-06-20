use std::fs;
use std::path::Path;

use command_tools::{CommandTools, run};

const GENERATE_GRAMMAR: Option<&str> = option_env!("GENERATE_GRAMMAR");

fn main() {
    generate_grammar();
}

fn generate_grammar() {
    if !cfg!(debug_assertions) && GENERATE_GRAMMAR.is_none_or(|x| x != "1") {
        return;
    }

    if needs_rebuild(
        "tree-sitter-nasin/src/parser.c",
        [
            "tree-sitter-nasin/grammar.js",
            "tree-sitter-nasin/package.json",
        ],
    ) {
        run!("bun", "install"; "tree-sitter-nasin");
        run!("bun", "tree-sitter", "generate"; "tree-sitter-nasin");
        run!("bun", "tree-sitter", "build"; "tree-sitter-nasin");
    }

    println!("cargo:rerun-if-changed=tree-sitter-nasin/grammar.js");
    println!("cargo:rerun-if-changed=tree-sitter-nasin/package.json");
}

fn needs_rebuild(
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
