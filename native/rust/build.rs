fn main() {
    let mut c_build = cc::Build::new();
    c_build.file("c/terlan_ndarray.c");
    c_build.file("c/terlan_ndarray_operations.c");
    c_build.include(".").warnings_into_errors(true).flag_if_supported("-std=c11").compile("terlan_native_boundary_c_abi");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=c");
}
