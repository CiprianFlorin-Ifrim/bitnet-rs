// Direct inference example for BitNet b1.58.
//
// Uses session.generate() which returns the complete response as a String,
// as opposed to the streaming examples which use callbacks.
//
// Usage:
//   cargo run --release --example inference_standard -- <path-to-gguf> "<prompt>"
//
// Example:
//   cargo run --release --example inference_standard -- \
//     models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf \
//     "What is the capital of France?"

use std::env;
use std::time::Instant;

use bitnet::{ContextParams, GenerateParams, Model, ModelParams, SamplingStrategy};

fn main() -> Result<(), bitnet::Error> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: inference_direct <path-to-gguf> <prompt>");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo run --release --example inference_direct -- \\");
        eprintln!("    models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf \\");
        eprintln!("    \"What is the capital of France?\"");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let prompt = &args[2];

    bitnet::init();
    bitnet::suppress_warnings();

    println!("Loading model...");
    let load_start = Instant::now();
    let model = Model::load(model_path, ModelParams::default())?;
    println!("Model loaded in {:.2}s\n", load_start.elapsed().as_secs_f32());

    let mut session = model.session(ContextParams::default())?;

    let formatted = format!(
        "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    );

    let params = GenerateParams {
        max_tokens: 200,
        sampling: SamplingStrategy::TopP {
            temperature: 0.8,
            top_p: 0.95,
            seed: u32::MAX,
        },
    };

    println!("Prompt: {prompt}\n");

    let gen_start = Instant::now();
    let response = session.generate(&formatted, &params)?;
    let elapsed = gen_start.elapsed().as_secs_f32();

    println!("--- Response ---\n");
    println!("{response}");
    println!("\n--- Stats ---");
    println!("Time             : {elapsed:.2}s");
    println!("Tokens per second: {:.2}", response.split_whitespace().count() as f32 / elapsed);

    bitnet::deinit();
    Ok(())
}