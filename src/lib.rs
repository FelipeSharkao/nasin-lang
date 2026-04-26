#![allow(irrefutable_let_patterns)]

use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::{fs, io, process};

use clap::Args;

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
use self::config::BuildConfig;
use self::errors::CompilerError;
use self::utils::cmd;

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
    /// Whether to dump the parsed bytecode of the source file after transformations (e.g.
    /// monomorphization)
    dump_transformed_bytecode: bool,
    #[arg(long)]
    /// Whether to dump the parsed bytecode of the source file before type inference and
    /// type checking is performed
    dump_untyped_bytecode: bool,
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
    let mut path_erros = vec![];

    let lib_dirs = option_env!("LIB_DIR")
        .iter()
        .flat_map(|s| s.split(':'))
        .filter_map(|x| match PathBuf::from(x).canonicalize() {
            Ok(path) => Some(path),
            Err(err) => {
                path_erros.push(errors::Error::new(
                    errors::ReadError::new(x.into(), err.kind()).into(),
                    None,
                ));
                None
            }
        })
        .collect();

    if !path_erros.is_empty() {
        return Err(CompilerError::new(None, path_erros));
    }

    let file = match emit.file.canonicalize() {
        Ok(file) => file,
        Err(err) => {
            path_erros.push(errors::Error::new(
                errors::ReadError::new(emit.file.clone(), err.kind()).into(),
                None,
            ));
            return Err(CompilerError::new(None, path_erros));
        }
    };

    let base_dir = match file.parent() {
        Some(parent) => parent.to_owned(),
        None => {
            path_erros.push(errors::Error::new(
                errors::ReadError::new(file.clone(), io::ErrorKind::IsADirectory).into(),
                None,
            ));
            return Err(CompilerError::new(None, path_erros));
        }
    };

    let name = file.file_stem().unwrap();

    let mut ctx = context::BuildContext::new(BuildConfig {
        name: name.to_string_lossy().to_string(),
        out: out.unwrap_or_else(|| base_dir.join(name)),
        base_dir,
        lib_dirs,
        silent: emit.silent,
        dump_ast: emit.dump_ast,
        dump_bytecode: emit.dump_bytecode,
        dump_transformed_bytecode: emit.dump_transformed_bytecode,
        dump_untyped_bytecode: emit.dump_untyped_bytecode,
        dump_clif: emit.dump_clif,
        run,
    });

    let Ok(src_idx) = ctx.open(file) else {
        return Err(ctx.into_compile_error());
    };

    if !ctx.parse_library() {
        return Err(ctx.into_compile_error());
    }

    ctx.parse(src_idx);
    if ctx.has_errors() {
        if ctx.cfg.dump_bytecode || ctx.cfg.dump_transformed_bytecode {
            b::Printer::new(&ctx.lock_modules(), &ctx.cfg)
                .with_show_ids(true)
                .with_source_manager(&ctx.source_manager)
                .print_all();
        }

        return Err(ctx.into_compile_error());
    }

    if ctx.cfg.dump_bytecode {
        b::Printer::new(&ctx.lock_modules(), &ctx.cfg)
            .with_show_ids(true)
            .with_source_manager(&ctx.source_manager)
            .print_all();
    }

    let code_transform = transform::CodeTransform::new(&ctx);
    code_transform.apply(transform::InstantiateGenericFuncsStep::new(&ctx));
    code_transform.apply(transform::LowerTypeNameStep::new(&ctx));
    code_transform.apply(transform::FinishGetPropertyStep::new(&ctx));
    code_transform.apply(transform::FinishDispatchStep::new(&ctx));

    if ctx.cfg.dump_transformed_bytecode {
        b::Printer::new(&ctx.lock_modules(), &ctx.cfg)
            .with_show_ids(true)
            .with_source_manager(&ctx.source_manager)
            .print_all();
    }

    ctx.compile();
    if ctx.has_errors() {
        return Err(ctx.into_compile_error());
    }

    if ctx.cfg.run {
        let status = cmd!(ctx.cfg.out).status().unwrap();
        if let Some(code) = status.code() {
            process::exit(code);
        } else if let Some(signal) = status.signal() {
            if signal == 11 {
                eprintln!("Segmentation fault");
                eprintln!(
                    "Unless you are doing some unsafe stuff, this is likely a bug in Nasin itself"
                );
            } else {
                eprintln!("Terminated by signal {signal}");
            }
            process::exit(128 + signal as i32);
        } else {
            process::exit(1);
        }
    }

    Ok(())
}
