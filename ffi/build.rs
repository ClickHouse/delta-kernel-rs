extern crate cbindgen;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::{env, fs};

use cbindgen::{Config, Language};

fn get_target_dir(manifest_dir: &str) -> PathBuf {
    if let Ok(target) = env::var("CARGO_TARGET_DIR") {
        // allow a CARGO_TARGET_DIR var that could be set via e.g. cmake to override where we put
        // the header files
        PathBuf::from(target)
    } else {
        let mut manifest_dir = PathBuf::from(manifest_dir);
        manifest_dir.pop(); // go up, since we're a sub-crate
        manifest_dir.join("target").join("ffi-headers")
    }
}

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR should be set");
    let package_name = env::var("CARGO_PKG_NAME").expect("CARGO_PKG_NAME should be set");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR should be set");
    let target_dir = get_target_dir(crate_dir.as_str());
    let cbindgen_toml = Path::new(&crate_dir).join("cbindgen.toml");
    let mut config = Config::from_file(&cbindgen_toml)
        .unwrap_or_else(|_| panic!("Couldn't find {}", cbindgen_toml.display()));

    // generate cxx bindings
    let output_file_hpp = target_dir.join(format!("{package_name}.hpp"));
    let mut config_hpp = config.clone();
    config_hpp.language = Language::Cxx;
    let bindings_hpp = cbindgen::generate_with_config(&crate_dir, config_hpp)
        .expect("generate_with_config should have worked for Cxx");
    bindings_hpp.write_to_file(&output_file_hpp);

    // generate c bindings
    let output_file_h = target_dir.join(format!("{package_name}.h"));
    config.language = Language::C;
    cbindgen::generate_with_config(&crate_dir, config)
        .expect("generate_with_config should have worked for C")
        .write_to_file(&output_file_h);

    // A `rerun-if` line turns off Cargo's default of watching this package, so list what the
    // headers come from: their directory, the files cbindgen read, the manifests of their packages,
    // and the workspace's Cargo.toml and Cargo.lock, from which its `cargo metadata` selects them.
    // The headers themselves are not listed: Cargo compares the paths with the start of the run.
    println!("cargo::rerun-if-env-changed=CARGO_TARGET_DIR");
    let depfile = PathBuf::from(out_dir).join(format!("{package_name}.d"));
    bindings_hpp.generate_depfile(&output_file_hpp, &depfile);
    let workspace_dir = Path::new(&crate_dir)
        .parent()
        .expect("the crate should be in a workspace");
    let mut inputs = BTreeSet::from([
        workspace_dir.join("Cargo.toml"),
        workspace_dir.join("Cargo.lock"),
    ]);
    // `<header>: \` then one path per line, spaces escaped as `\ `.
    let depfile_contents =
        fs::read_to_string(&depfile).expect("cbindgen should have written the depfile");
    for line in depfile_contents.lines().skip(1) {
        let source = PathBuf::from(
            line.trim()
                .trim_end_matches('\\')
                .trim_end()
                .replace("\\ ", " "),
        );
        if let Some(package_dir) = source
            .ancestors()
            .skip(1)
            .find(|dir| dir.join("Cargo.toml").is_file())
        {
            inputs.insert(package_dir.join("Cargo.toml"));
        }
        inputs.insert(source);
    }
    for input in inputs.iter().filter(|input| input.is_file()) {
        println!("cargo::rerun-if-changed={}", input.display());
    }
}
