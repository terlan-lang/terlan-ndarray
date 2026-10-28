fn main() {
    let mut probe = pkg_config::Config::new();
    let static_openblas = std::env::var("OPENBLAS_STATIC").as_deref() == Ok("1");
    probe.statik(static_openblas);
    probe.atleast_version("0.3.0");
    let library = probe
        .probe("openblas")
        .expect("pkg-config must resolve external C package openblas");
    let mut c_build = cc::Build::new();
    c_build.file("c/terlan_ndarray.c");
    c_build.file("c/terlan_ndarray_order.c");
    c_build.file("c/terlan_ndarray_output.c");
    c_build.file("c/terlan_ndarray_dtype.c");
    c_build.file("c/terlan_ndarray_creation.c");
    c_build.file("c/terlan_ndarray_creation_extra.c");
    c_build.file("c/terlan_ndarray_elementwise.c");
    c_build.file("c/terlan_ndarray_scalar_math.c");
    c_build.file("c/terlan_ndarray_stride_tricks.c");
    c_build.file("c/terlan_ndarray_unary.c");
    c_build.file("c/terlan_ndarray_ufunc_output.c");
    c_build.file("c/terlan_ndarray_indexing.c");
    c_build.file("c/terlan_ndarray_advanced_indexing.c");
    c_build.file("c/terlan_ndarray_shape.c");
    c_build.file("c/terlan_ndarray_shape_extra.c");
    c_build.file("c/terlan_ndarray_completion.c");
    c_build.file("c/terlan_ndarray_padding.c");
    c_build.file("c/terlan_ndarray_ordering.c");
    c_build.file("c/terlan_ndarray_set.c");
    c_build.file("c/terlan_ndarray_operations.c");
    c_build.file("c/terlan_ndarray_linear_algebra.c");
    c_build.file("c/terlan_ndarray_reductions.c");
    c_build.file("c/terlan_ndarray_reduction_options.c");
    c_build.file("c/terlan_ndarray_noncommutative.c");
    c_build.file("c/terlan_ndarray_statistics.c");
    c_build.file("c/terlan_ndarray_covariance.c");
    c_build.file("c/terlan_ndarray_histogram.c");
    c_build.file("c/terlan_ndarray_numerical.c");
    c_build.file("c/terlan_ndarray_gradient.c");
    c_build.file("c/terlan_ndarray_decomposition.c");
    c_build.file("c/terlan_ndarray_dlpack.c");
    for include_path in &library.include_paths {
        c_build.include(include_path);
    }
    c_build
        .include("include")
        .include(".")
        .warnings_into_errors(true)
        .flag_if_supported("-std=c11")
        .compile("terlan_native_boundary_c_abi");
    println!("cargo:rerun-if-changed=include");
    println!("cargo:rerun-if-changed=c");
    println!("cargo:rerun-if-env-changed=OPENBLAS_STATIC");
}
