use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let bitnet_src = manifest_dir.join("bitnet.cpp");

    if !bitnet_src.exists() {
        panic!(
            "\n\nbitnet.cpp source not found at {bitnet_src:?}.\n\
             Please run:\n\
             \n  git submodule add https://github.com/microsoft/BitNet bitnet-sys/bitnet.cpp\n\
             \n  git submodule update --init --recursive\n\n"
        );
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    let mut config = cmake::Config::new(&bitnet_src);

    config
        .define("CMAKE_BUILD_TYPE", "Release")
        // Static linking avoids shipping additional shared libraries alongside the binary.
        .define("BUILD_SHARED_LIBS", "OFF")
        // These targets are not needed as we are only linking the library itself.
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        // The install step in this fork fails because LlamaConfig.cmake is never
        // generated. Skipping the install step entirely avoids that failure while
        // still producing the compiled archives we need.
        .no_build_target(true);

    // BitNet's efficiency comes from lookup-table kernels that replace floating point
    // multiply-accumulate with integer table lookups. The correct kernel must be selected
    // at compile time based on the target architecture, otherwise the model falls back
    // to standard matrix multiplication which is both slower and produces incorrect
    // output with the i2_s weight format.
    //
    // TL1 is the ARM variant, used on Apple Silicon.
    // TL2 is the x86 variant targeting AVX-512, with automatic fallback to TL1/AVX2
    // if the host CPU does not support AVX-512.
    match (target_arch.as_str(), target_os.as_str()) {
        ("aarch64", "macos") => {
            config.define("LLAMA_BITNET_ARM_TL1", "ON");
            config.define("GGML_METAL", "OFF");
        }
        ("x86_64", _) => {
            config.define("LLAMA_BITNET_X86_TL2", "ON");
        }
        _ => {}
    }

    let dst = config.build();

    // The cmake install step copies libllama.a and libggml.a to out/lib before
    // failing on the missing LlamaConfig.cmake. Those copies are authoritative
    // so we search there first, then fall back to the raw cmake build tree in
    // case a particular archive was not copied.
    let lib_dir = dst.join("lib");
    let build_dir = dst.join("build");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-search=native={}/3rdparty/llama.cpp/src", build_dir.display());
    println!("cargo:rustc-link-search=native={}/3rdparty/llama.cpp/ggml/src", build_dir.display());

    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");

    match target_os.as_str() {
        "macos" => {
            // Foundation and Accelerate are required by ggml on macOS regardless of
            // whether Metal is enabled. Accelerate provides the BLAS implementation
            // that ggml uses for non-BitNet tensor operations.
            //
            // Metal and MetalKit must be linked even in CPU-only mode because ggml
            // always compiles the Metal backend on macOS and references Metal symbols
            // at link time.
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=c++");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=m");
        }
        _ => {}
    }

    println!("cargo:rerun-if-changed=bitnet.cpp/CMakeLists.txt");
    println!("cargo:rerun-if-changed=bitnet.cpp/include/llama.h");
    println!("cargo:rerun-if-changed=build.rs");
}