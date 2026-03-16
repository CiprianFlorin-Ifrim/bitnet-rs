//! Safe Rust bindings to Microsoft's BitNet b1.58 inference engine.
//!
//! # Single-turn usage
//!
//! ```no_run
//! use bitnet::{init, suppress_warnings, Model, ModelParams, ContextParams, GenerateParams};
//!
//! init();
//! suppress_warnings();
//!
//! let model = Model::load(
//!     "models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf",
//!     ModelParams::default(),
//! )?;
//! let mut session = model.session(ContextParams::default())?;
//! let response = session.generate("The capital of France is", &GenerateParams::default())?;
//! println!("{response}");
//! # Ok::<(), bitnet::Error>(())
//! ```
//!
//! # Multi-turn chat
//!
//! Sessions track the KV cache position across turns. Only new tokens are
//! encoded on each turn — history is never re-encoded.
//!
//! ```no_run
//! use bitnet::{init, suppress_warnings, Model, ModelParams, ContextParams, GenerateParams};
//!
//! init();
//! suppress_warnings();
//!
//! let model = Model::load(
//!     "models/bitnet-b1.58-2B-4T-bf16/ggml-model-i2s-bitnet.gguf",
//!     ModelParams::default(),
//! )?;
//! let mut session = model.session(ContextParams::default())?;
//!
//! // Turn 1 — BOS added automatically because kv_pos is 0
//! session.generate_streaming(
//!     "<|start_header_id|>system<|end_header_id|>\nYou are helpful.<|eot_id|>\
//!      <|start_header_id|>user<|end_header_id|>\nHello!<|eot_id|>\
//!      <|start_header_id|>assistant<|end_header_id|>\n",
//!     &GenerateParams::default(),
//!     |piece| print!("{piece}"),
//! )?;
//! session.encode("<|eot_id|>")?;
//!
//! // Turn 2 — only new tokens, no re-encoding of history
//! session.generate_streaming(
//!     "<|start_header_id|>user<|end_header_id|>\nHow are you?<|eot_id|>\
//!      <|start_header_id|>assistant<|end_header_id|>\n",
//!     &GenerateParams::default(),
//!     |piece| print!("{piece}"),
//! )?;
//! session.encode("<|eot_id|>")?;
//! # Ok::<(), bitnet::Error>(())
//! ```
//!
//! # Model setup
//!
//! The GGUF file must be produced from the BF16 weights using the conversion
//! helper in the bitnet.cpp repository. The pre-packaged GGUF from Hugging Face
//! is missing pre-tokenizer metadata and produces incoherent output.
//!
//! ```sh
//! hf download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir models/bitnet-b1.58-2B-4T-bf16
//! python utils/convert-helper-bitnet.py models/bitnet-b1.58-2B-4T-bf16
//! ```
//!
//! # CPU only
//!
//! BitNet uses lookup-table kernels for its ternary weight format that only
//! run on the CPU. GPU offloading produces incorrect output and is not
//! supported by this crate.

mod error;
mod model;
mod params;
mod session;

pub use error::Error;
pub use model::Model;
pub use params::{ContextParams, GenerateParams, ModelParams, SamplingStrategy};
pub use session::Session;

/// Initialises the ggml backend. Must be called once before the first
/// Model::load call.
pub fn init() {
    unsafe { bitnet_llm_sys::llama_backend_init() };
}

/// Releases resources allocated by init. Optional at process exit.
pub fn deinit() {
    unsafe { bitnet_llm_sys::llama_backend_free() };
}

/// Suppresses warning-level output from the underlying library while keeping
/// informational messages such as memory usage and context size.
pub fn suppress_warnings() {
    unsafe extern "C" fn filter_log(
        _level: std::ffi::c_int,
        text: *const std::ffi::c_char,
        _user_data: *mut std::ffi::c_void,
    ) {
        let s = unsafe { std::ffi::CStr::from_ptr(text) };
        if let Ok(s) = s.to_str() {
            if s.contains("pre-tokenizer")
                || s.contains("GENERATION QUALITY")
                || s.contains("CONSIDER REGENERATING")
                || s.contains("****")
                || s.contains("is not marked as EOG")
                || s.contains("special_eos_id is not in special_eog_ids")
                || s.contains("control token:")
            {
                return;
            }
            eprint!("{s}");
        }
    }
    unsafe { bitnet_llm_sys::llama_log_set(Some(filter_log), std::ptr::null_mut()) };
}