use std::ffi::CString;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bitnet_llm_sys as sys;

use crate::error::Error;
use crate::model::ModelInner;
use crate::params::{ContextParams, GenerateParams, SamplingStrategy};

struct SessionInner {
    ctx: NonNull<sys::llama_context>,
    _model: Arc<ModelInner>,
}

unsafe impl Send for SessionInner {}

impl Drop for SessionInner {
    fn drop(&mut self) {
        unsafe { sys::llama_free(self.ctx.as_ptr()) };
    }
}

/// An inference context tied to a loaded model.
///
/// Tracks KV cache position across calls, allowing incremental encoding for
/// multi-turn conversations without re-encoding history on every turn.
///
/// # Single-turn usage
///
/// ```no_run
/// use bitnet::{Model, ModelParams, ContextParams, GenerateParams};
///
/// let model = Model::load("model.gguf", ModelParams::default())?;
/// let mut session = model.session(ContextParams::default())?;
/// let response = session.generate("Hello!", &GenerateParams::default())?;
/// println!("{response}");
/// # Ok::<(), bitnet::Error>(())
/// ```
///
/// # Multi-turn usage
///
/// Sessions track the KV cache position across turns. Only new tokens are
/// encoded on each turn — history is never re-encoded. You must call
/// `encode("<|eot_id|>")` after each assistant turn to close it correctly,
/// and call `reset()` between separate conversations.
///
/// BOS is added automatically on the first call when `kv_pos == 0`. Do not
/// call `encode` before `generate_streaming` on a fresh session or BOS will
/// be skipped.
///
/// ```no_run
/// use bitnet::{Model, ModelParams, ContextParams, GenerateParams};
///
/// let model = Model::load("model.gguf", ModelParams::default())?;
/// let mut session = model.session(ContextParams::default())?;
///
/// // Turn 1 — BOS added automatically because kv_pos is 0
/// session.generate_streaming(
///     "<|start_header_id|>system<|end_header_id|>\nYou are helpful.<|eot_id|>\
///      <|start_header_id|>user<|end_header_id|>\nHello!<|eot_id|>\
///      <|start_header_id|>assistant<|end_header_id|>\n",
///     &GenerateParams::default(),
///     |piece| print!("{piece}"),
/// )?;
/// session.encode("<|eot_id|>")?;
///
/// // Turn 2 — only new tokens encoded, history stays in KV cache
/// session.generate_streaming(
///     "<|start_header_id|>user<|end_header_id|>\nHow are you?<|eot_id|>\
///      <|start_header_id|>assistant<|end_header_id|>\n",
///     &GenerateParams::default(),
///     |piece| print!("{piece}"),
/// )?;
/// session.encode("<|eot_id|>")?;
///
/// // Start a fresh conversation on the same session
/// session.reset();
/// # Ok::<(), bitnet::Error>(())
/// ```
pub struct Session {
    inner: SessionInner,
    /// Current end position of the KV cache. Incremented after every token
    /// fed to the model, whether during prefill, encode, or generation.
    kv_pos: usize,
    /// Maximum context size in tokens, cached from the context at creation.
    n_ctx: usize,
}

impl Session {
    pub(crate) fn new(model: Arc<ModelInner>, params: ContextParams) -> Result<Self, Error> {
        let threads = params.n_threads.max(1) as i32;

        let mut c_params = unsafe { sys::llama_context_default_params() };
        c_params.n_ctx = params.n_ctx;
        c_params.n_batch = params.n_batch;
        // n_ubatch must equal n_batch so the library never splits our batches
        // internally. Internal splitting of multi-token batches through the
        // BitNet lookup-table kernel produces corrupted output.
        c_params.n_ubatch = params.n_batch;
        c_params.n_threads = threads;
        c_params.n_threads_batch = threads;

        let raw = unsafe {
            sys::llama_new_context_with_model(model.ptr.as_ptr(), c_params)
        };

        let ctx = NonNull::new(raw).ok_or_else(|| {
            Error::ContextCreate("llama_new_context_with_model returned null".into())
        })?;

        let n_ctx = unsafe { sys::llama_n_ctx(ctx.as_ptr()) } as usize;

        Ok(Self {
            inner: SessionInner { ctx, _model: model },
            kv_pos: 0,
            n_ctx,
        })
    }

    /// Clears the KV cache and resets the position counter.
    ///
    /// Call this between separate conversations to reuse the session without
    /// paying the cost of creating a new context.
    pub fn reset(&mut self) {
        unsafe { sys::llama_kv_cache_clear(self.inner.ctx.as_ptr()) };
        self.kv_pos = 0;
    }

    /// Returns the current KV cache position in tokens.
    pub fn kv_pos(&self) -> usize {
        self.kv_pos
    }

    /// Returns the total context window size in tokens.
    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// Returns the number of tokens remaining before the context window is full.
    pub fn tokens_remaining(&self) -> usize {
        self.n_ctx.saturating_sub(self.kv_pos)
    }

    /// Encodes text into the KV cache without generating any output.
    ///
    /// Used to feed tokens that should not trigger generation — typically the
    /// closing `<|eot_id|>` after each assistant turn in a multi-turn
    /// conversation. Does not add a BOS token.
    ///
    /// Do not call this before the first `generate_streaming` call on a fresh
    /// session — doing so will prevent BOS from being added.
    pub fn encode(&mut self, text: &str) -> Result<(), Error> {
        let ctx = self.inner.ctx.as_ptr();
        let model = self.inner._model.ptr.as_ptr();

        if self.kv_pos + 1 >= self.n_ctx {
            return Err(Error::KvCacheFull);
        }

        let tokens = tokenise(model, text, false)?;
        if tokens.is_empty() {
            return Ok(());
        }

        for (i, &tok) in tokens.iter().enumerate() {
            let mut single = [tok];
            let pos = (self.kv_pos + i) as i32;
            let batch = unsafe {
                sys::llama_batch_get_one(single.as_mut_ptr(), 1, pos, 0)
            };
            match unsafe { sys::llama_decode(ctx, batch) } {
                0 => {}
                1 => return Err(Error::KvCacheFull),
                n => return Err(Error::Decode(n)),
            }
        }
        self.kv_pos += tokens.len();
        Ok(())
    }

    /// Runs inference on the prompt and returns the complete generated text.
    ///
    /// Note: for multi-turn usage, prefer `generate_streaming` and call
    /// `encode("<|eot_id|>")` afterward to properly close the assistant turn
    /// in the KV cache.
    pub fn generate(&mut self, prompt: &str, params: &GenerateParams) -> Result<String, Error> {
        let mut output = String::new();
        self.generate_streaming(prompt, params, |piece| output.push_str(piece))?;
        Ok(output)
    }

    /// Runs inference on the prompt, invoking the callback for each decoded
    /// token piece as it is produced.
    ///
    /// BOS is added automatically when `kv_pos == 0`. Subsequent calls do not
    /// add BOS, allowing incremental encoding of conversation turns.
    ///
    /// Generation stops when the model emits an end-of-sequence token or when
    /// `max_tokens` tokens have been produced.
    ///
    /// After this returns, call `encode("<|eot_id|>")` to close the assistant
    /// turn before the next user message.
    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        params: &GenerateParams,
        mut on_token: impl FnMut(&str),
    ) -> Result<(), Error> {
        let ctx = self.inner.ctx.as_ptr();
        let model = self.inner._model.ptr.as_ptr();

        // Add BOS only at the very start of a fresh conversation.
        let add_bos = self.kv_pos == 0;
        let tokens = tokenise(model, prompt, add_bos)?;

        // Warn when approaching the context limit.
        let tokens_needed = tokens.len() + params.max_tokens;
        if self.kv_pos + tokens_needed >= self.n_ctx {
            return Err(Error::Tokenise(format!(
                "context window full: {} used + {} needed >= {} max. Call reset() to start a new conversation.",
                self.kv_pos, tokens_needed, self.n_ctx
            )));
        }

        // Warn when over 80% full so callers can act before hitting the limit.
        if self.kv_pos as f32 / self.n_ctx as f32 > 0.8 {
            eprintln!(
                "warning: context window {:.0}% full ({}/{} tokens used)",
                100.0 * self.kv_pos as f32 / self.n_ctx as f32,
                self.kv_pos,
                self.n_ctx
            );
        }

        let sampler = build_sampler(&params.sampling)?;

        // Feed each prompt token individually at the correct KV cache position.
        // The BitNet lookup-table kernel only produces valid results with
        // single-token batches. The incremental KV cache means only NEW tokens
        // are processed each turn so this is fast regardless.
        for (i, &tok) in tokens.iter().enumerate() {
            let mut single = [tok];
            let pos = (self.kv_pos + i) as i32;
            let batch = unsafe {
                sys::llama_batch_get_one(single.as_mut_ptr(), 1, pos, 0)
            };
            match unsafe { sys::llama_decode(ctx, batch) } {
                0 => {}
                1 => {
                    unsafe { sys::llama_sampler_free(sampler) };
                    return Err(Error::KvCacheFull);
                }
                n => {
                    unsafe { sys::llama_sampler_free(sampler) };
                    return Err(Error::Decode(n));
                }
            }
        }
        self.kv_pos += tokens.len();

        let mut next_token = [0i32; 1];
        let mut n_generated = 0usize;

        loop {
            if n_generated >= params.max_tokens {
                break;
            }

            let token = unsafe { sys::llama_sampler_sample(sampler, ctx, 0) };
            unsafe { sys::llama_sampler_accept(sampler, token) };

            if unsafe { sys::llama_token_is_eog(model, token) } {
                break;
            }

            let piece = token_to_text(model, token)?;
            on_token(&piece);
            n_generated += 1;

            next_token[0] = token;
            let batch = unsafe {
                sys::llama_batch_get_one(next_token.as_mut_ptr(), 1, self.kv_pos as i32, 0)
            };
            match unsafe { sys::llama_decode(ctx, batch) } {
                0 => {}
                1 => {
                    unsafe { sys::llama_sampler_free(sampler) };
                    return Err(Error::KvCacheFull);
                }
                n => {
                    unsafe { sys::llama_sampler_free(sampler) };
                    return Err(Error::Decode(n));
                }
            }
            self.kv_pos += 1;
        }

        unsafe { sys::llama_sampler_free(sampler) };
        Ok(())
    }
}

fn tokenise(
    model: *const sys::llama_model,
    text: &str,
    add_special: bool,
) -> Result<Vec<sys::llama_token>, Error> {
    let c_text = CString::new(text)
        .map_err(|_| Error::Tokenise("prompt contains a null byte".into()))?;

    let n_required = unsafe {
        sys::llama_tokenize(
            model,
            c_text.as_ptr(),
            text.len() as i32,
            std::ptr::null_mut(),
            0,
            add_special,
            true,
        )
    };

    if n_required == 0 {
        return Err(Error::Tokenise("prompt produced zero tokens".into()));
    }

    let capacity = n_required.unsigned_abs() as usize;
    let mut buf: Vec<sys::llama_token> = vec![0; capacity];

    let n_tokens = unsafe {
        sys::llama_tokenize(
            model,
            c_text.as_ptr(),
            text.len() as i32,
            buf.as_mut_ptr(),
            capacity as i32,
            add_special,
            true,
        )
    };

    if n_tokens < 0 {
        return Err(Error::Tokenise(format!(
            "tokenisation failed with code {n_tokens}"
        )));
    }

    buf.truncate(n_tokens as usize);
    Ok(buf)
}

fn token_to_text(
    model: *const sys::llama_model,
    token: sys::llama_token,
) -> Result<String, Error> {
    let mut buf = vec![0u8; 32];
    loop {
        let n = unsafe {
            sys::llama_token_to_piece(
                model,
                token,
                buf.as_mut_ptr() as *mut std::ffi::c_char,
                buf.len() as i32,
                0,
                false,
            )
        };
        if n < 0 {
            buf.resize((-n) as usize, 0);
        } else {
            buf.truncate(n as usize);
            return String::from_utf8(buf).map_err(|e| {
                Error::Internal(format!("token {token} produced invalid UTF-8: {e}"))
            });
        }
    }
}

fn build_sampler(strategy: &SamplingStrategy) -> Result<*mut sys::llama_sampler, Error> {
    let chain_params = unsafe { sys::llama_sampler_chain_default_params() };
    let chain = unsafe { sys::llama_sampler_chain_init(chain_params) };

    if chain.is_null() {
        return Err(Error::Internal("failed to create sampler chain".into()));
    }

    match strategy {
        SamplingStrategy::Greedy => {
            let stage = unsafe { sys::llama_sampler_init_greedy() };
            unsafe { sys::llama_sampler_chain_add(chain, stage) };
        }
        SamplingStrategy::TopP { temperature, top_p, seed } => {
            let penalty_stage = unsafe {
                sys::llama_sampler_init_penalties(64, 1.1, 0.0, 0.0)
            };
            unsafe { sys::llama_sampler_chain_add(chain, penalty_stage) };

            let top_k_stage = unsafe { sys::llama_sampler_init_top_k(40) };
            unsafe { sys::llama_sampler_chain_add(chain, top_k_stage) };

            if *top_p < 1.0 {
                let top_p_stage = unsafe { sys::llama_sampler_init_top_p(*top_p, 1) };
                unsafe { sys::llama_sampler_chain_add(chain, top_p_stage) };
            }

            let min_p_stage = unsafe { sys::llama_sampler_init_min_p(0.05, 1) };
            unsafe { sys::llama_sampler_chain_add(chain, min_p_stage) };

            let temp_stage = unsafe { sys::llama_sampler_init_temp(*temperature) };
            unsafe { sys::llama_sampler_chain_add(chain, temp_stage) };

            let effective_seed = if *seed == u32::MAX {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(42)
            } else {
                *seed
            };

            let dist_stage = unsafe { sys::llama_sampler_init_dist(effective_seed) };
            unsafe { sys::llama_sampler_chain_add(chain, dist_stage) };
        }
    }

    Ok(chain)
}