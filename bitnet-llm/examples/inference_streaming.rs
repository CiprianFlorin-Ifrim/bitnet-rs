// Single-prompt inference example for BitNet b1.58.
//
// Takes a prompt, runs inference once, prints the response and timing stats.
//
// Usage:
//   cargo run --release --example inference_streaming -- <path-to-gguf> "<prompt>"
//
// Example:
//   cargo run --release --example inference_streaming -- \
//     models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf \
//     "The capital of France is"

use std::env;
use std::io::Write;
use std::time::Instant;

use bitnet_llm::{ContextParams, GenerateParams, Model, ModelParams, SamplingStrategy};

fn main() -> Result<(), bitnet_llm::Error> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: inference <path-to-gguf> <prompt>");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo run --release --example inference -- \\");
        eprintln!("    models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf \\");
        eprintln!("    \"The capital of France is\"");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let prompt = &args[2];

    bitnet_llm::init();
    bitnet_llm::suppress_warnings();

    println!("Loading model from {model_path}...");
    let load_start = Instant::now();
    let model = Model::load(model_path, ModelParams::default())?;
    println!("Model loaded in {:.2}s\n", load_start.elapsed().as_secs_f32());

    let mut session = model.session(ContextParams::default())?;

    let params = GenerateParams {
        max_tokens: 200,
        sampling: SamplingStrategy::TopP {
            temperature: 0.8,
            top_p: 0.95,
            seed: u32::MAX,
        },
    };

    // Wrap the prompt in the Llama 3 chat template the model was trained with.
    let formatted = format!(
        "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    );

    println!("Prompt: {prompt}\n");
    println!("--- Response ---\n");

    let mut token_count = 0usize;
    let gen_start = Instant::now();

    session.generate_streaming(&formatted, &params, |piece| {
        print!("{piece}");
        let _ = std::io::stdout().flush();
        token_count += 1;
    })?;

    let elapsed = gen_start.elapsed().as_secs_f32();

    println!("\n\n--- Stats ---");
    println!("Tokens generated : {token_count}");
    println!("Time             : {elapsed:.2}s");
    println!("Tokens per second: {:.2}", token_count as f32 / elapsed);

    bitnet_llm::deinit();
    Ok(())
}