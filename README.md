# bitnet-llm

Safe Rust bindings to Microsoft's [BitNet b1.58](https://github.com/microsoft/BitNet)
inference engine (`bitnet.cpp`).

Provides a clean, ergonomic API with both full-response and streaming inference,
multi-turn chat with incremental KV cache, and targeting macOS (Apple Silicon)
and Linux (x86-64).

---

## Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
bitnet-llm = "1.0.0"
```

Note: the first build compiles bitnet.cpp from source via CMake which takes a
few minutes. Subsequent builds use the cached output.

---

## Prerequisites

| Tool | Notes |
|------|-------|
| Rust >= 1.73 | `rustup update stable` |
| CMake >= 3.14 | `brew install cmake` / `apt install cmake` |
| Clang | Required by bitnet.cpp; usually already present |
| Python 3 + pip | For model conversion only, not at Rust build time |
| `hf` CLI | `pip install huggingface_hub` |

**macOS (Apple Silicon) only:** Xcode Command Line Tools (`xcode-select --install`).

---

## Model setup

The pre-packaged GGUF on Hugging Face is missing pre-tokenizer metadata and
produces incoherent output. You must convert from the BF16 master weights.
```sh
# Create a conda environment (recommended)
conda create -n bitnet python=3.11
conda activate bitnet

# Install conversion dependencies
pip install huggingface_hub numpy torch

# Download the BF16 master weights (~5 GB)
hf download microsoft/bitnet-b1.58-2B-4T-bf16 \
    --local-dir models/bitnet-b1.58-2B-4T-bf16

# Clone the BitNet repository for the conversion script
git clone https://github.com/microsoft/BitNet /tmp/bitnet
pip install -r /tmp/bitnet/requirements.txt

# Convert to GGUF format
python /tmp/bitnet/utils/convert-helper-bitnet.py \
    models/bitnet-b1.58-2B-4T-bf16
```

The output file is `models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf`.

---

## Build

### macOS (Apple Silicon)
```sh
cargo build --release
```

### Linux (x86-64)
```sh
cargo build --release
```

The build script auto-detects AVX-512 and enables the `TL2` BitNet kernel if
available, falling back to the `TL1` kernel automatically. Build time is a few
minutes on first run as bitnet.cpp is compiled from source. Subsequent builds
use the cached CMake output.

---

## Usage

### Running the examples
```sh
# Single prompt streaming inference
cargo run --release --example inference_streaming -- \
    models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf \
    "What is the capital of France?"

# Single prompt, returns full response at once
cargo run --release --example inference_standard -- \
    models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf \
    "What is the capital of France?"

# Interactive multi-turn chat
cargo run --release --example chat -- \
    models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf \
    "You are a helpful assistant."
```

### Library usage

#### Single-turn inference
```rust
use bitnet_llm::{
    init, suppress_warnings, ContextParams, GenerateParams,
    Model, ModelParams, SamplingStrategy,
};
use std::io::Write;

fn main() -> Result<(), bitnet_llm::Error> {
    init();
    suppress_warnings();

    let model = Model::load(
        "models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf",
        ModelParams::default(),
    )?;

    let mut session = model.session(ContextParams::default())?;

    let params = GenerateParams {
        max_tokens: 200,
        sampling: SamplingStrategy::TopP {
            temperature: 0.8,
            top_p: 0.95,
            seed: u32::MAX,
        },
    };

    // Streaming — callback fires for each token piece as it is produced
    session.generate_streaming(
        "<|start_header_id|>user<|end_header_id|>\n\
         What is the capital of France?<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n",
        &params,
        |piece| {
            print!("{piece}");
            let _ = std::io::stdout().flush();
        },
    )?;

    // Or collect the full response at once
    let response = session.generate(
        "<|start_header_id|>user<|end_header_id|>\n\
         What is the capital of France?<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n",
        &params,
    )?;
    println!("{response}");

    bitnet_llm::deinit();
    Ok(())
}
```

#### Multi-turn chat

Sessions track KV cache position across turns. Only new tokens are encoded on
each turn — history is never re-encoded. This keeps subsequent turns fast
regardless of conversation length.
```rust
use bitnet_llm::{
    init, suppress_warnings, ContextParams, GenerateParams,
    Model, ModelParams, SamplingStrategy,
};
use std::io::Write;

fn main() -> Result<(), bitnet_llm::Error> {
    init();
    suppress_warnings();

    let model = Model::load(
        "models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf",
        ModelParams::default(),
    )?;

    let mut session = model.session(ContextParams::default())?;
    let params = GenerateParams::default();

    // Turn 1 — BOS added automatically because kv_pos is 0
    session.generate_streaming(
        "<|start_header_id|>system<|end_header_id|>\n\
         You are a helpful assistant.<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\
         Hello!<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n",
        &params,
        |piece| { print!("{piece}"); let _ = std::io::stdout().flush(); },
    )?;
    // Close the assistant turn in the KV cache
    session.encode("<|eot_id|>")?;

    // Turn 2 — only new tokens encoded, full history stays in KV cache
    session.generate_streaming(
        "<|start_header_id|>user<|end_header_id|>\n\
         How are you?<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n",
        &params,
        |piece| { print!("{piece}"); let _ = std::io::stdout().flush(); },
    )?;
    session.encode("<|eot_id|>")?;

    // Start a fresh conversation on the same session
    session.reset();

    bitnet_llm::deinit();
    Ok(())
}
```

---

## API overview

| Item | Description |
|------|-------------|
| `init()` | Initialise the backend. Call once before `Model::load`. |
| `deinit()` | Free backend resources. Optional at process exit. |
| `suppress_warnings()` | Suppress tokenizer noise while keeping memory and context info. |
| `Model::load(path, ModelParams)` | Load a GGUF model from disk. |
| `model.session(ContextParams)` | Create an inference context. |
| `session.generate(prompt, &params)` | Run inference, return `String`. |
| `session.generate_streaming(prompt, &params, callback)` | Run inference, call callback per token piece. |
| `session.encode(text)` | Feed text into the KV cache without generating output. |
| `session.reset()` | Clear the KV cache and start a fresh conversation. |
| `session.kv_pos()` | Current token position in the KV cache. |
| `session.tokens_remaining()` | Tokens remaining before context window is full. |

### `ModelParams`

| Field | Default | Description |
|-------|---------|-------------|
| `n_gpu_layers` | `0` | Must be 0. BitNet kernels are CPU-only. |
| `use_mmap` | `true` | Memory-mapped weight loading. |
| `use_mlock` | `false` | Lock weights in RAM. |

### `ContextParams`

| Field | Default | Description |
|-------|---------|-------------|
| `n_ctx` | `0` | Context window size. 0 uses the model maximum (4096). |
| `n_batch` | `32` | Internal batch buffer size. |
| `n_threads` | auto | CPU threads for inference. |

### `GenerateParams`

| Field | Default | Description |
|-------|---------|-------------|
| `max_tokens` | `512` | Maximum tokens to generate. |
| `sampling` | `TopP` | `Greedy` or `TopP { temperature, top_p, seed }`. |

---

## Important notes

**CPU only.** BitNet's ternary lookup-table kernels only run on the CPU. The
ARM TL1 kernel is used on Apple Silicon and the x86 TL2 kernel on Linux. GPU
offloading is not supported.

**Single-token decode loop.** The BitNet kernels produce valid output only when
decoding one token at a time for multi-turn systems. 
This library handles this correctly internally.

**Chat template.** This model uses the Llama 3 chat template with
`<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>` control tokens.
See the examples for complete implementations.

**Context window.** The model has a 4096 token context window. The library
warns at 80% usage and returns an error when full. Call `session.reset()` to
start a fresh conversation.

**Multi-turn correctness.** Always call `session.encode("<|eot_id|>")` after
each assistant turn. Always call `session.reset()` between separate
conversations. Do not call `session.encode()` before the first
`generate_streaming` call on a fresh session as this will prevent BOS from
being added.