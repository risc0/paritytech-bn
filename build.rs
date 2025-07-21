fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let vend = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();

    println!("cargo::rustc-check-cfg=cfg(target_r0vm)");
    if os == "zkvm" && vend == "risc0" {
        println!("cargo:rustc-cfg=target_r0vm");
    }
}
