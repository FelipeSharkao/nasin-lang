#!/usr/bin/env -S cargo +nightly -Zscript
---
[package]
edition = "2024"
[dependencies]
clap = { version = "4.6.1", features = ["derive"] }
command-tools = { path = "command-tools" }
scopeguard = "1.2.0"
---

use std::path::Path;
use std::{env, fs};

use clap::Parser;
use command_tools::{CommandTools, cmd};
use scopeguard::defer;

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    run: bool,
    #[arg(short, long)]
    test: bool,
    #[arg(long)]
    record: bool,
    #[arg(long)]
    release: bool,
    #[arg(long)]
    generate_grammar: bool,
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

fn main() {
    let cli = Cli::parse();

    generate_grammar(&cli);

    cmd!(
        "cargo",
        if cli.run { "run" } else { "build" },
        if cli.release { Some("--release") } else { None },
        "--",
        &cli.args,
    )
    .env("RUST_BACKTRACE", "1")
    .run();

    if !cli.run {
        if !fs::exists("bin").unwrap() {
            fs::create_dir("bin").unwrap();
        }

        if cli.release {
            fs::copy("target/release/nasin", "bin/nasin").unwrap();
        } else {
            fs::copy("target/debug/nasin", "bin/nasin").unwrap();
        }
    }

    if cli.test {
        cmd!("./rere.py", "replay", "tests/_test.list").run();
    }

    if cli.record {
        cmd!("./rere.py", "record", "tests/_test.list").run();
    }
}

fn generate_grammar(cli: &Cli) {
    if !cfg!(debug_assertions) && cli.generate_grammar {
        return;
    }

    if needs_rebuild(
        "tree-sitter-nasin/src/parser.c",
        [
            "tree-sitter-nasin/grammar.js",
            "tree-sitter-nasin/package.json",
        ],
    ) {
        let cwd = env::current_dir().unwrap();
        env::set_current_dir("tree-sitter-nasin").unwrap();
        defer!(env::set_current_dir(cwd).unwrap());

        cmd!("bun", "install").run();
        cmd!("bun", "tree-sitter", "generate").run();
        cmd!("bun", "tree-sitter", "build").run();
    }
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
