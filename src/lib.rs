#![allow(irrefutable_let_patterns)]

use std::collections::HashSet;
use std::path::PathBuf;
use std::{env, fs, io};

use clap::Args;
use command_tools::{CommandTools, cmd};
use itertools::chain;

mod bytecode;
mod codegen;
mod config;
mod context;
mod errors;
mod parser;
mod sources;
mod transform;
mod typecheck;
mod utils;

use self::bytecode as b;
use self::config::{BuildConfig, ShouldDump};
use self::errors::CompilerError;

#[derive(Args, Debug)]
pub struct EmitArgs {
    /// Path to the file to compile
    file: PathBuf,
    #[arg(long, short)]
    /// Omit all messages
    silent: bool,
    #[arg(long)]
    /// Whether to dump the AST of the source file
    dump_ast: Option<Option<String>>,
    #[arg(long)]
    /// Whether to dump the parsed bytecode of the source file
    dump_bytecode: Option<Option<String>>,
    #[arg(long)]
    /// Whether to dump the parsed bytecode of the source file after transformations (e.g.
    /// monomorphization)
    dump_transformed_bytecode: Option<Option<String>>,
    #[arg(long)]
    /// Whether to dump the parsed bytecode of the source file before type inference and
    /// type checking is performed
    dump_untyped_bytecode: Option<Option<String>>,
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
    let file = match emit.file.canonicalize() {
        Ok(file) => file,
        Err(err) => {
            let error = errors::Error::new(
                errors::ReadError::new(emit.file.clone(), err.kind()).into(),
                None,
            );
            return Err(CompilerError::new(None, HashSet::from([error])));
        }
    };

    let base_dir = match file.parent() {
        Some(parent) => parent.to_owned(),
        None => {
            let error = errors::Error::new(
                errors::ReadError::new(file.clone(), io::ErrorKind::IsADirectory).into(),
                None,
            );
            return Err(CompilerError::new(None, HashSet::from([error])));
        }
    };

    let name = file.file_stem().unwrap();

    let mut ctx = context::BuildContext::new(BuildConfig {
        name: name.to_string_lossy().to_string(),
        out: out.unwrap_or_else(|| base_dir.join(name)),
        base_dir,
        lib_dirs: get_lib_dirs(),
        silent: emit.silent,
        dump_ast: emit.dump_ast.into(),
        dump_bytecode: emit.dump_bytecode.into(),
        dump_transformed_bytecode: emit.dump_transformed_bytecode.into(),
        dump_untyped_bytecode: emit.dump_untyped_bytecode.into(),
        dump_clif: emit.dump_clif,
        run,
    });

    ctx.parse_library();

    let Ok(src_idx) = ctx.open(file) else {
        return Err(ctx.into_compile_error());
    };

    ctx.parse(src_idx);
    ctx.parse_runtime();

    if ctx.has_errors() {
        let flag = [&ctx.cfg.dump_bytecode, &ctx.cfg.dump_transformed_bytecode];
        if !flag.never_dumps() {
            b::Printer::new(&ctx.lock_modules(), &ctx.cfg)
                .with_show_ids(true)
                .with_source_manager(&ctx.source_manager)
                .print(flag.as_slice());
        }

        return Err(ctx.into_compile_error());
    }

    if !ctx.cfg.dump_bytecode.never_dumps() {
        b::Printer::new(&ctx.lock_modules(), &ctx.cfg)
            .with_show_ids(true)
            .with_source_manager(&ctx.source_manager)
            .print(&ctx.cfg.dump_bytecode);
    }

    let mut code_transform = transform::CodeTransform::new(&ctx);
    code_transform.apply(transform::InstantiateGenericFuncsStep::new(&ctx));
    code_transform.apply(transform::LowerTypeNameStep::new(&ctx));
    code_transform.apply(transform::FinishGetPropertyStep::new(&ctx));
    code_transform.apply(transform::FinishDispatchStep::new(&ctx));

    if !ctx.cfg.dump_transformed_bytecode.never_dumps() {
        b::Printer::new(&ctx.lock_modules(), &ctx.cfg)
            .with_show_ids(true)
            .with_source_manager(&ctx.source_manager)
            .print(&ctx.cfg.dump_transformed_bytecode);
    }

    ctx.compile();
    if ctx.has_errors() {
        return Err(ctx.into_compile_error());
    }

    if ctx.cfg.run {
        if let Err(err) = cmd!(ctx.cfg.out).exec() {
            eprintln!("{}: {err}", ctx.cfg.name);
            err.exit_process();
        }
    }

    Ok(())
}

fn get_lib_dirs() -> Vec<PathBuf> {
    chain!(
        (|| Some(env::current_dir().ok()?.join("libs")))(),
        data_home_dir().map(|p| p.join("nasin/libs")),
        local_share_dir().map(|p| p.join("nasin/libs")),
    )
    .collect()
}

fn data_home_dir() -> Option<PathBuf> {
    if env::consts::OS == "windows" {
        env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| Some(env::home_dir()?.join(".local/share")))
    }
}

fn local_share_dir() -> Option<PathBuf> {
    if env::consts::OS == "windows" {
        env::var_os("PROGRAMFILES")
            .map(PathBuf::from)
            .or_else(|| Some(PathBuf::from("C:/Program Files")))
    } else {
        Some(PathBuf::from("/usr/local/share"))
    }
}
