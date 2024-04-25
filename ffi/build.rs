extern crate cbindgen;

use cbindgen::{Config, ExportConfig, Language, MangleConfig};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

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
    let target_dir = get_target_dir(crate_dir.as_str());

    // Any `cfgs` we want to turn into ifdefs need to go in here
    let defines: HashMap<String, String> = HashMap::from([(
        "feature = default-client".into(),
        "DEFINE_DEFAULT_CLIENT".into(),
    )]);

    // generate cxx bindings
    let output_file_hpp = target_dir
        .join(format!("{}.hpp", package_name))
        .display()
        .to_string();
    let mut config_hpp = Config::default();
    config_hpp.language = Language::Cxx;
    config_hpp.namespace = Some(String::from("ffi"));
    config_hpp.defines = defines.clone();
    cbindgen::generate_with_config(&crate_dir, config_hpp)
        .expect("generate_with_config should have worked for Cxx")
        .write_to_file(output_file_hpp);

    // generate c bindings
    let output_file_h = target_dir
        .join(format!("{}.h", package_name))
        .display()
        .to_string();
    let mut config_h = Config::default();
    config_h.language = Language::C;
    config_h.defines = defines;
    let mut mangle_config = MangleConfig::default();
    mangle_config.remove_underscores = true;
    let mut export_config = ExportConfig::default();
    export_config.mangle = mangle_config;
    config_h.export = export_config;
    cbindgen::generate_with_config(&crate_dir, config_h)
        .expect("generate_with_config should have worked for C")
        .write_to_file(output_file_h);
}
