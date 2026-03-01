#![recursion_limit = "256"]

//! CPUT Calendar Q&A System
//!
//! A complete Question-Answering pipeline built with Rust and the Burn
//! deep learning framework (v0.20.1). Reads CPUT institutional calendar
//! Word documents (.docx) and answers natural language questions.
//!
//! ## Usage
//!
//! ```bash
//! # Train the model
//! cargo run -- train ./data --epochs 10 --lr 0.0001 --batch-size 4
//!
//! # Answer a question using a saved checkpoint
//! cargo run -- ask model_checkpoint.json "When does Term 1 start in 2026?"
//!
//! # Run a demo (train + sample Q&A)
//! cargo run -- demo ./data
//! ```

mod data;
mod inference;
mod model;
mod tokenizer;
mod training;

use std::env;

fn print_usage() {
    println!("CPUT Calendar Q&A System");
    println!();
    println!("USAGE:");
    println!("  word-doc-qa train <data_dir> [OPTIONS]");
    println!("  word-doc-qa ask <checkpoint.json> <question>");
    println!("  word-doc-qa demo <data_dir>");
    println!();
    println!("OPTIONS (for train):");
    println!("  --epochs <n>       Number of training epochs (default: 10)");
    println!("  --lr <f>           Learning rate (default: 0.0001)");
    println!("  --batch-size <n>   Batch size (default: 4)");
    println!("  --output <path>    Checkpoint output path (default: model_checkpoint.json)");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "train" => cmd_train(&args[2..]),
        "ask" => cmd_ask(&args[2..]),
        "demo" => cmd_demo(&args[2..]),
        "--help" | "-h" | "help" => print_usage(),
        other => {
            eprintln!("Unknown command: '{}'. Run with --help for usage.", other);
            std::process::exit(1);
        }
    }
}

/// `train` subcommand
fn cmd_train(args: &[String]) {
    if args.is_empty() {
        eprintln!("Error: data_dir is required. Example: cargo run -- train ./data");
        std::process::exit(1);
    }

    let data_dir = args[0].clone();
    let mut epochs = 10usize;
    let mut lr = 1e-4f64;
    let mut batch_size = 4usize;
    let mut output = "model_checkpoint.json".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--epochs" if i + 1 < args.len() => {
                epochs = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --epochs value; using default 10");
                    10
                });
                i += 2;
            }
            "--lr" if i + 1 < args.len() => {
                lr = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --lr value; using default 0.0001");
                    1e-4
                });
                i += 2;
            }
            "--batch-size" if i + 1 < args.len() => {
                batch_size = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --batch-size value; using default 4");
                    4
                });
                i += 2;
            }
            "--output" if i + 1 < args.len() => {
                output = args[i + 1].clone();
                i += 2;
            }
            flag => {
                eprintln!("Unknown flag '{}'; ignoring.", flag);
                i += 1;
            }
        }
    }

    println!("=== CPUT Calendar Q&A — Training ===");
    println!("  data_dir   : {}", data_dir);
    println!("  epochs     : {}", epochs);
    println!("  lr         : {}", lr);
    println!("  batch_size : {}", batch_size);
    println!("  output     : {}", output);

    let config = training::TrainingConfig {
        epochs,
        learning_rate: lr,
        batch_size,
        data_dir,
        model_output: output,
    };

    match training::train(config) {
        Ok(()) => println!("\n✓ Training complete!"),
        Err(e) => {
            eprintln!("\n✗ Training failed: {}", e);
            std::process::exit(1);
        }
    }
}

/// `ask` subcommand
fn cmd_ask(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Error: checkpoint path and question required.");
        eprintln!("Example: cargo run -- ask model_checkpoint.json \"When is Good Friday?\"");
        std::process::exit(1);
    }

    let checkpoint_path = &args[0];
    let question = args[1..].join(" ");

    println!("=== CPUT Calendar Q&A — Inference ===");
    println!("Checkpoint : {}", checkpoint_path);
    println!("Question   : {}", question);
    println!();

    match inference::answer_question(checkpoint_path, &question) {
        Ok(answer) => {
            println!("Answer: {}", answer);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// `demo` subcommand — trains a small model then shows sample answers
fn cmd_demo(args: &[String]) {
    if args.is_empty() {
        eprintln!("Error: data_dir required for demo.");
        std::process::exit(1);
    }

    let data_dir = &args[0];
    println!("=== CPUT Calendar Q&A — Demo Mode ===");

    // Quick 5-epoch training run
    let config = training::TrainingConfig {
        epochs: 5,
        learning_rate: 1e-4,
        batch_size: 4,
        data_dir: data_dir.clone(),
        model_output: "demo_checkpoint.json".to_string(),
    };

    println!("\n--- Phase 1: Training (5 epochs) ---");
    if let Err(e) = training::train(config) {
        eprintln!("Training error: {}", e);
        std::process::exit(1);
    }

    println!("\n--- Phase 2: Q&A Demo ---");
    let demo_questions = [
        "What is the date of the 2026 End of Year Graduation Ceremony?",
        "When does Term 1 start in 2026?",
        "How many times did the HDC hold their meetings in 2024?",
        "When is Good Friday in 2026?",
        "When is the Research Festival in 2026?",
        "When does the academic year start for Academic Staff in 2026?",
        "When does Term 4 end in 2026?",
        "When is Heritage Day in 2026?",
    ];

    for (i, q) in demo_questions.iter().enumerate() {
        println!("\nQ{}: {}", i + 1, q);
        match inference::answer_question("demo_checkpoint.json", q) {
            Ok(ans) => println!("A{}: {}", i + 1, ans),
            Err(e) => println!("A{}: [Error] {}", i + 1, e),
        }
    }

    println!("\n=== Demo complete ===");
}
