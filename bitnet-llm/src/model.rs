use std::ffi::CString;
use std::ptr::NonNull;
use std::sync::Arc;

use bitnet_llm_sys as sys;

use crate::error::Error;
use crate::params::{ContextParams, ModelParams};
use crate::session::Session;

pub(crate) struct ModelInner {
    pub(crate) ptr: NonNull<sys::llama_model>,
}

unsafe impl Send for ModelInner {}
unsafe impl Sync for ModelInner {}

impl Drop for ModelInner {
    fn drop(&mut self) {
        unsafe { sys::llama_free_model(self.ptr.as_ptr()) };
    }
}

#[derive(Clone)]
pub struct Model {
    pub(crate) inner: Arc<ModelInner>,
}

impl Model {
    pub fn load(path: impl AsRef<std::path::Path>, params: ModelParams) -> Result<Self, Error> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| Error::ModelLoad("path contains non-UTF-8 characters".into()))?;

        let c_path = CString::new(path_str)
            .map_err(|_| Error::ModelLoad("path contains a null byte".into()))?;

        let mut c_params = unsafe { sys::llama_model_default_params() };
        c_params.n_gpu_layers = params.n_gpu_layers;
        c_params.use_mlock = params.use_mlock;
        c_params.use_mmap = params.use_mmap;
        c_params.check_tensors = false;

        let raw = unsafe { sys::llama_load_model_from_file(c_path.as_ptr(), c_params) };

        let ptr = NonNull::new(raw).ok_or_else(|| {
            Error::ModelLoad(format!("failed to load model from {path_str:?}"))
        })?;

        Ok(Self {
            inner: Arc::new(ModelInner { ptr }),
        })
    }

    pub fn session(&self, params: ContextParams) -> Result<Session, Error> {
        Session::new(Arc::clone(&self.inner), params)
    }
}