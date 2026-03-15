/// Parameters for loading a model from disk.
#[derive(Debug, Clone)]
pub struct ModelParams {
    /// Number of transformer layers to offload to a GPU backend.
    ///
    /// Must be 0 for BitNet. The i2_s ternary weight format relies on CPU
    /// lookup-table kernels that have no GPU equivalent. Setting this above 0
    /// causes affected layers to fall back to standard matrix multiplication,
    /// producing incorrect output.
    pub n_gpu_layers: i32,

    /// Locks model weights in RAM after loading, preventing the OS from
    /// swapping them out. Recommended for repeated inference on
    /// memory-constrained systems.
    pub use_mlock: bool,

    /// Uses memory-mapped I/O to load weights. Faster than a standard read on
    /// most filesystems and allows the OS to reclaim pages under memory
    /// pressure. Only disable if your filesystem does not support mmap.
    pub use_mmap: bool,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            use_mlock: false,
            use_mmap: true,
        }
    }
}

/// Parameters for creating an inference context from a loaded model.
#[derive(Debug, Clone)]
pub struct ContextParams {
    /// The context window size in tokens. 0 uses the model's trained maximum,
    /// which is 4096 for BitNet b1.58 2B4T.
    pub n_ctx: u32,

    /// Batch size for token processing. This MUST be 1 for BitNet. The ARM
    /// TL1 and x86 TL2 lookup-table kernels that give BitNet its efficiency
    /// are designed for single-token batches. Multi-token batches cause the
    /// kernels to produce NaN logits, resulting in garbage output.
    pub n_batch: u32,

    /// Number of CPU threads for inference. Defaults to the number of logical
    /// CPUs available. On Apple Silicon, setting this to the number of
    /// performance cores (4 on M4) gives the best results.
    pub n_threads: u32,
}

impl Default for ContextParams {
    fn default() -> Self {
        // Use all available CPUs but cap at 8 to avoid scheduling overhead
        // on machines with many efficiency cores. On M4 MacBook Air this
        // resolves to 10, but 4 performance cores handle inference better.
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4)
            .min(8);
        Self {
            n_ctx: 0,
            n_batch: 32,
            n_threads: cpus,
        }
    }
}

/// The token sampling strategy used during generation.
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Always selects the highest probability token. Deterministic but tends
    /// toward repetition on longer generations.
    Greedy,

    /// Samples from the filtered distribution after temperature scaling and
    /// nucleus (top-p) filtering.
    TopP {
        /// Scales the logit distribution before sampling. Lower values make
        /// output more focused. Higher values increase creativity. Typical
        /// range is 0.1 to 1.5.
        temperature: f32,

        /// Discards tokens whose cumulative probability exceeds this threshold
        /// before sampling. 1.0 disables top-p filtering.
        top_p: f32,

        /// Random seed. u32::MAX derives a seed from the system clock, giving
        /// different output on each call. Fixed values give reproducible results.
        seed: u32,
    },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        SamplingStrategy::TopP {
            temperature: 0.7,
            top_p: 0.9,
            seed: u32::MAX,
        }
    }
}

/// Parameters controlling a single generation call.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    /// Maximum number of tokens to generate. Generation may stop earlier if
    /// the model produces an end-of-sequence token.
    pub max_tokens: usize,

    /// The sampling strategy to use when selecting each output token.
    pub sampling: SamplingStrategy,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            sampling: SamplingStrategy::default(),
        }
    }
}