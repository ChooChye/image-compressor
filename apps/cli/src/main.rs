//! Command-line wrapper around `smartimg-core`.
//!
//! - `smartimg <input>`: smallest encoding meeting an SSIM target (research mode).
//! - `smartimg web <inputs...>`: web-ready AVIF/WebP at fixed quality under a byte budget.

mod optimize;
mod web;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "smartimg", version, args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    optimize: optimize::Args,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Resize and encode images for websites: AVIF + WebP fallback under a size cap.
    Web(web::Args),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Some(Command::Web(args)) => web::run(args),
        None => optimize::run(cli.optimize),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}
