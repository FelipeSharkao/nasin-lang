use std::io::IsTerminal;
use std::path::PathBuf;
use std::str::FromStr;
use std::{env, fs, io, process};

use clap::{Parser, Subcommand};
use nasin::{build, build_run, EmitArgs};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Parser, Debug)]
#[command(name = "Nasin Language")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: CliCommand,
}

#[derive(Subcommand, Debug)]
enum CliCommand {
    #[clap(alias = "b")]
    /// Build a source file
    Build {
        #[arg(long, short)]
        /// Path where to place the output file
        out:  Option<PathBuf>,
        #[command(flatten)]
        emit: EmitArgs,
    },
    #[clap(alias = "r")]
    /// Build and run a source file
    Run {
        #[command(flatten)]
        emit: EmitArgs,
    },
}

fn main() {
    //unsafe { compact_debug::enable(true) };

    prepare_tracing();

    let cli = Cli::parse();

    let result = match cli.cmd {
        CliCommand::Build { out, emit } => build(emit, out),
        CliCommand::Run { emit } => build_run(emit),
    };

    if let Err(error) = result {
        eprintln!("{}", error);
        process::exit(1);
    }
}

fn prepare_tracing() {
    macro_rules! level {
        () => {
            tracing_subscriber::fmt::layer().with_level(true).pretty()
        };
    }

    let file = env::var("LOG_FILE").ok().map(|log_file| {
        let path = PathBuf::from_str(log_file.as_ref())
            .expect("LOG_FILE should be a valid path");
        fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .expect("LOG_FILE should be writable")
    });

    tracing_subscriber::registry()
        .with(if file.is_none() {
            Some(
                level!()
                    .with_ansi(io::stderr().is_terminal())
                    .with_writer(io::stderr),
            )
        } else {
            None
        })
        .with(file.map(|file| level!().with_ansi(false).with_writer(file)))
        .with(match env::var("LOG_LEVEL") {
            Ok(s) => match s.as_str() {
                "trace" => LevelFilter::TRACE,
                "debug" => LevelFilter::DEBUG,
                "info" => LevelFilter::INFO,
                _ => LevelFilter::WARN,
            },
            _ => LevelFilter::WARN,
        })
        .with(filter_fn(|meta| meta.target().starts_with("nasin")))
        .init();
}
