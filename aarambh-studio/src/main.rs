//! Command-line interface for training, tuning, evaluating, and serving Aarambh AI models.

#![deny(missing_docs)]

mod cmd;
mod ui;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "aarambh-studio")]
#[command(version)]
#[command(about = "Aarambh AI command line tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Agent(Box<cmd::agent::AgentArgs>),
    Train(cmd::train::TrainArgs),
    Infer(Box<cmd::infer::InferArgs>),
    Eval(Box<cmd::eval::EvalArgs>),
    Quantise(cmd::quantise::QuantiseArgs),
    Convert(cmd::convert::ConvertArgs),
    Distill(Box<cmd::distill::DistillArgs>),
    Finetune(Box<cmd::finetune::FinetuneArgs>),
    Merge(cmd::merge::MergeArgs),
    Selflearn(Box<cmd::selflearn::SelflearnArgs>),
    Serve(cmd::serve::ServeArgs),
    Retrieve(cmd::retrieve::RetrieveArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Agent(args) => cmd::agent::run(*args),
        Command::Train(args) => cmd::train::run(args),
        Command::Infer(args) => cmd::infer::run(*args),
        Command::Eval(args) => cmd::eval::run(*args),
        Command::Quantise(args) => cmd::quantise::run(args),
        Command::Convert(args) => cmd::convert::run(args),
        Command::Distill(args) => cmd::distill::run(*args),
        Command::Finetune(args) => cmd::finetune::run(*args),
        Command::Merge(args) => cmd::merge::run(args),
        Command::Selflearn(args) => cmd::selflearn::run(*args),
        Command::Serve(args) => cmd::serve::run(args),
        Command::Retrieve(args) => cmd::retrieve::run(args),
    }
}
