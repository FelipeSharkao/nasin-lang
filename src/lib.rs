#![allow(irrefutable_let_patterns)]

use std::path::PathBuf;
use std::sync::Arc;
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
    });

    ctx.parse_library();
    let src_idx = ctx.preload(emit.file).expect("file not found");

    ctx.parse(src_idx);

    if { ctx.errors.lock().unwrap() }.len() > 0 {
        let source_manager = Arc::new(ctx.source_manager);
        let errors = ctx.errors.into_inner().unwrap();
        return Err(CompilerError::new(source_manager, errors));
    }

    ctx.compile();
    Ok(())
}

pub fn build_run(emit: EmitArgs) -> Result<(), CompilerError> {
    let build_dir = emit.file.parent().unwrap().join("build");
    fs::create_dir_all(&build_dir).unwrap();

    let out = build_dir.join(emit.file.file_stem().unwrap());
    build(emit, Some(out.clone()))?;

    let status = cmd!(out).status().unwrap();
    process::exit(status.code().unwrap_or(1));
}
