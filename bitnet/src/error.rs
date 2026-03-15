use std::fmt;

/// Errors that can be returned by this crate.
#[derive(Debug)]
pub enum Error {
    /// The GGUF file could not be loaded. This is usually a bad path, an
    /// unsupported format version, or insufficient memory to map the file.
    ModelLoad(String),

    /// The inference context could not be created. The most common cause is
    /// requesting a context size larger than available memory.
    ContextCreate(String),

    /// The prompt could not be tokenised. This typically means the prompt
    /// contains a null byte or exceeds the context window size.
    Tokenise(String),

    /// llama_decode returned a non-zero status code. The integer is the raw
    /// return value from the C function for diagnostic purposes.
    Decode(i32),

    /// The KV cache ran out of space during generation. Either the prompt is
    /// too long or max_tokens is set too high for the configured context size.
    KvCacheFull,

    /// An internal invariant was violated. This should never happen in normal
    /// use and indicates a bug in the crate. Please file an issue.
    Internal(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ModelLoad(msg)     => write!(f, "failed to load model: {msg}"),
            Error::ContextCreate(msg) => write!(f, "failed to create context: {msg}"),
            Error::Tokenise(msg)      => write!(f, "tokenisation error: {msg}"),
            Error::Decode(code)       => write!(f, "llama_decode failed with code {code}"),
            Error::KvCacheFull        => write!(f, "KV cache is full; reduce max_tokens or shorten the prompt"),
            Error::Internal(msg)      => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}