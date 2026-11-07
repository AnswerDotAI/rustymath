//! The cfgs that select the linear algebra. `accelerate` holds on macOS, where Accelerate serves products and decompositions. `native` holds where rustymath's own kernels are compiled: on every other target, and on macOS with the `native` feature.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(accelerate, native)");
    let mac = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos");
    if mac { println!("cargo::rustc-cfg=accelerate") }
    if !mac || std::env::var_os("CARGO_FEATURE_NATIVE").is_some() { println!("cargo::rustc-cfg=native") }
}
