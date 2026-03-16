// Multi-turn interactive chat example for BitNet b1.58.
//
// Uses incremental KV cache encoding — only new tokens are processed on each
// turn, history is never re-encoded. This eliminates the prefill delay on
// subsequent turns.
//
// Usage:
//   cargo run --release --example inference_chat_multiturn -- <path-to-gguf>
//   cargo run --release --example inference_chat_multiturn -- <path-to-gguf> "system prompt"

use std::env;
use std::io::{self, BufRead, Write};
use std::time::Instant;

use bitnet_llm::{ContextParams, GenerateParams, Model, ModelParams, SamplingStrategy};

fn main() -> Result<(), bitnet_llm::Error> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: inference_chat_multiturn <path-to-gguf> [system-prompt]");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo run --release --example inference_chat_multiturn -- \\");
        eprintln!("    models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf \\");
        eprintln!("    \"You are a helpful assistant.\"");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let system_prompt = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("You are a helpful assistant.");

    bitnet_llm::init();
    bitnet_llm::suppress_warnings();

    println!("Loading model from {model_path}...");
    let load_start = Instant::now();
    let model = Model::load(model_path, ModelParams::default())?;
    println!("Model loaded in {:.2}s\n", load_start.elapsed().as_secs_f32());
    println!("System: {system_prompt}");
    println!("Type your message and press Enter. Type 'quit' to exit.\n");

    let params = GenerateParams {
        max_tokens: 512,
        sampling: SamplingStrategy::TopP {
            temperature: 0.8,
            top_p: 0.95,
            seed: u32::MAX,
        },
    };

    let mut session = model.session(ContextParams::default())?;
    let stdin = io::stdin();
    let mut first_turn = true;

    loop {
        print!("You: ");
        let _ = io::stdout().flush();

        let mut user_input = String::new();
        if stdin.lock().read_line(&mut user_input).is_err() {
            break;
        }
        let user_input = user_input.trim().to_string();

        if user_input.is_empty() {
            continue;
        }
        if user_input.eq_ignore_ascii_case("quit")
            || user_input.eq_ignore_ascii_case("exit")
        {
            break;
        }

        // On the first turn include the system prompt. On subsequent turns
        // only the new user message is sent — the session already has the
        // full history in its KV cache.
        let prompt = if first_turn {
            format!(
                "<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\
                 <|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n"
            )
        } else {
            format!(
                "<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n"
            )
        };

        print!("Assistant: ");
        let _ = io::stdout().flush();

        let mut token_count = 0usize;
        let mut ttft: Option<f32> = None;
        let gen_start = Instant::now();

        session.generate_streaming(&prompt, &params, |piece| {
            if ttft.is_none() {
                ttft = Some(gen_start.elapsed().as_secs_f32());
            }
            print!("{piece}");
            let _ = io::stdout().flush();
            token_count += 1;
        })?;

        // Encode the closing token so the KV cache reflects the complete turn.
        session.encode("<|eot_id|>")?;

        let elapsed = gen_start.elapsed().as_secs_f32();
        println!(
            "\n[{token_count} tokens, ttft: {:.2}s, total: {elapsed:.2}s, {:.1} tok/s]\n",
            ttft.unwrap_or(0.0),
            token_count as f32 / elapsed
        );

        first_turn = false;
    }

    bitnet_llm::deinit();
    Ok(())
}