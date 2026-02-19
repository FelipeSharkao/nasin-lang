#![allow(irrefutable_let_patterns)]

use std::path::PathBuf;
use std::{fs, process};

use clap::Args;

use self::config::BuildConfig;
use self::errors::CompilerError;
use self::utils::cmd;

pub mod bytecode;
pub mod codegen;
pub mod config;
pub mod context;
pub mod errors;
pub mod parser;
pub mod sources;
pub mod typecheck;
mod utils;

#[derive(Args, Debug)]
pub struct EmitArgs {
    /// Path to the file to compile
    file: PathBuf,
    #[arg(long, short)]
    /// Omit all messages
    silent: bool,
    #[arg(long)]
    /// Whether to dump the AST of the source file
    dump_ast: bool,
    #[arg(long)]
    /// Whether to dump the parsed bytecode of the source file
    dump_bytecode: bool,
    #[arg(long)]
    /// Whether to dump the parsed CLIF of the source file, if using Cranelift
    dump_clif: bool,
}

pub fn build(emit: EmitArgs, out: Option<PathBuf>) -> Result<(), CompilerError> {
    build_maybe_run(emit, out, false)
}

pub fn build_run(emit: EmitArgs) -> Result<(), CompilerError> {
    let build_dir = emit.file.parent().unwrap().join("build");
    fs::create_dir_all(&build_dir).unwrap();

    let out = build_dir.join(emit.file.file_stem().unwrap());
    build_maybe_run(emit, Some(out), true)
}

pub fn build_maybe_run(
    emit: EmitArgs,
    out: Option<PathBuf>,
    run: bool,
) -> Result<(), CompilerError> {
    let mut ctx = context::BuildContext::new(BuildConfig {
        out: out.unwrap_or_else(|| {
            emit.file
                .parent()
                .unwrap()
                .join(emit.file.file_stem().unwrap())
        }),
        silent: emit.silent,
        dump_ast: emit.dump_ast,
        dump_bytecode: emit.dump_bytecode,
        dump_clif: emit.dump_clif,
        run,
    });

    if !ctx.parse_library() {
        return Err(ctx.into_compile_error());
    }

    let Ok(src_idx) = ctx.preload(emit.file) else {
        return Err(ctx.into_compile_error());
    };

    ctx.parse(src_idx, true);
    if ctx.has_errors() {
        return Err(ctx.into_compile_error());
    }

    ctx.compile();
    if ctx.has_errors() {
        return Err(ctx.into_compile_error());
    }

    if ctx.cfg.run {
        let status = cmd!(ctx.cfg.out).status().unwrap();
        process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}
