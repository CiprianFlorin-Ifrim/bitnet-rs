#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_char, c_float, c_int, c_void};

pub type llama_token = i32;
pub type llama_pos = i32;
pub type llama_seq_id = i32;

#[repr(C)]
pub struct llama_model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_sampler {
    _private: [u8; 0],
}

pub type llama_split_mode = i32;
pub type llama_rope_scaling_type = i32;
pub type llama_pooling_type = i32;
pub type llama_attention_type = i32;
pub type ggml_type = i32;

pub type llama_progress_callback =
    Option<unsafe extern "C" fn(progress: c_float, user_data: *mut c_void) -> bool>;

pub type ggml_backend_sched_eval_callback =
    Option<unsafe extern "C" fn(t: *mut c_void, ask: bool, user_data: *mut c_void) -> bool>;

pub type ggml_abort_callback =
    Option<unsafe extern "C" fn(data: *mut c_void) -> bool>;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_model_params {
    pub n_gpu_layers: c_int,
    pub split_mode: llama_split_mode,
    pub main_gpu: c_int,
    pub tensor_split: *const c_float,
    pub rpc_servers: *const c_char,
    pub progress_callback: llama_progress_callback,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const llama_model_kv_override,

    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,

    pub rope_scaling_type: llama_rope_scaling_type,
    pub pooling_type: llama_pooling_type,
    pub attention_type: llama_attention_type,

    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,

    pub cb_eval: ggml_backend_sched_eval_callback,
    pub cb_eval_user_data: *mut c_void,

    pub type_k: ggml_type,
    pub type_v: ggml_type,

    pub logits_all: bool,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub flash_attn: bool,
    pub no_perf: bool,

    _pad: [u8; 3],

    pub abort_callback: ggml_abort_callback,
    pub abort_callback_data: *mut c_void,
}

#[repr(C)]
pub struct llama_model_kv_override {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_sampler_chain_params {
    pub no_perf: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_batch {
    pub n_tokens: i32,

    pub token: *mut llama_token,
    pub embd: *mut f32,
    pub pos: *mut llama_pos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut llama_seq_id,
    pub logits: *mut i8,

    pub all_pos_0: llama_pos,
    pub all_pos_1: llama_pos,
    pub all_seq_id: llama_seq_id,

    _pad: [u8; 4],
}

unsafe extern "C" {
    pub fn llama_backend_init();
    pub fn llama_backend_free();

    pub fn llama_log_set(
        log_callback: Option<unsafe extern "C" fn(c_int, *const c_char, *mut c_void)>,
        user_data: *mut c_void,
    );

    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;

    pub fn llama_load_model_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;

    pub fn llama_free_model(model: *mut llama_model);

    pub fn llama_new_context_with_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;

    pub fn llama_free(ctx: *mut llama_context);
    pub fn llama_n_ctx(ctx: *const llama_context) -> u32;

    pub fn llama_tokenize(
        model: *const llama_model,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    pub fn llama_token_to_piece(
        model: *const llama_model,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;

    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> c_int;

    pub fn llama_batch_get_one(
        tokens: *mut llama_token,
        n_tokens: i32,
        pos_0: llama_pos,
        seq_id: llama_seq_id,
    ) -> llama_batch;

    pub fn llama_detokenize(
        model: *const llama_model,
        tokens: *const llama_token,
        n_tokens: i32,
        text: *mut c_char,
        text_len_max: i32,
        remove_special: bool,
        unparse_special: bool,
    ) -> i32;

    pub fn llama_kv_cache_clear(ctx: *mut llama_context);

    pub fn llama_sampler_chain_init(
        params: llama_sampler_chain_params,
    ) -> *mut llama_sampler;

    pub fn llama_sampler_chain_add(chain: *mut llama_sampler, smpl: *mut llama_sampler);
    pub fn llama_sampler_free(smpl: *mut llama_sampler);

    pub fn llama_sampler_init_greedy() -> *mut llama_sampler;
    pub fn llama_sampler_init_temp(t: c_float) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_k(k: i32) -> *mut llama_sampler;
    pub fn llama_sampler_init_min_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;
    pub fn llama_sampler_init_penalties(
        penalty_last_n: i32,
        penalty_repeat: c_float,
        penalty_freq: c_float,
        penalty_present: c_float,
    ) -> *mut llama_sampler;

    pub fn llama_sampler_sample(
        smpl: *mut llama_sampler,
        ctx: *mut llama_context,
        idx: i32,
    ) -> llama_token;

    pub fn llama_sampler_accept(smpl: *mut llama_sampler, token: llama_token);

    pub fn llama_token_bos(model: *const llama_model) -> llama_token;
    pub fn llama_token_eos(model: *const llama_model) -> llama_token;
    pub fn llama_token_is_eog(model: *const llama_model, token: llama_token) -> bool;
    pub fn llama_get_logits_ith(ctx: *mut llama_context, i: i32) -> *mut f32;
}