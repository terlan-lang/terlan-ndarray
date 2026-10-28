use super::*;

unsafe fn floats(array: *mut super::super::ffi::TerlanNdArray) -> Vec<f64> {
    let mut values = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the test owns a live floating array and releases the copied values.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_to_floats(array, &mut values, &mut length) },
        OK
    );
    // SAFETY: successful readback returns `length` initialized values.
    let result = unsafe { std::slice::from_raw_parts(values, length) }.to_vec();
    // SAFETY: the copied values use their matching package destructor.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_float_values(values) };
    result
}

unsafe fn integers(array: *mut super::super::ffi::TerlanNdArray) -> Vec<i64> {
    let mut values = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the test owns a live integer array and releases the copied values.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_ints_cast(array, &mut values, &mut length)
        },
        OK
    );
    // SAFETY: successful readback returns `length` initialized values.
    let result = unsafe { std::slice::from_raw_parts(values, length) }.to_vec();
    // SAFETY: the copied values use their matching package destructor.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_int_values(values) };
    result
}

unsafe fn dispose(array: *mut super::super::ffi::TerlanNdArray) {
    // SAFETY: each successful distribution call transfers one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(array) },
        OK
    );
}

#[test]
fn maintained_float_distributions_are_reproducible_and_bounded() {
    let shape = [128i64];
    let mut first_stream = std::ptr::null_mut();
    let mut second_stream = std::ptr::null_mut();
    let mut first = std::ptr::null_mut();
    let mut second = std::ptr::null_mut();
    // SAFETY: output slots are valid and disposed below.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(71, &mut first_stream) },
        OK
    );
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(71, &mut second_stream) },
        OK
    );
    // SAFETY: streams and borrowed shape are live for each call.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_beta(
                first_stream,
                shape.as_ptr(),
                1,
                2.0,
                5.0,
                3,
                &mut first,
            )
        },
        OK
    );
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_beta(
                second_stream,
                shape.as_ptr(),
                1,
                2.0,
                5.0,
                3,
                &mut second,
            )
        },
        OK
    );
    // SAFETY: both calls returned live Float64 arrays.
    let first_values = unsafe { floats(first) };
    let second_values = unsafe { floats(second) };
    assert_eq!(first_values, second_values);
    assert!(first_values.iter().all(|value| (0.0..=1.0).contains(value)));
    // SAFETY: every resource is live and uniquely owned.
    unsafe { dispose(first) };
    unsafe { dispose(second) };
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(first_stream) },
        OK
    );
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(second_stream) },
        OK
    );
}

#[test]
fn gamma_normal_and_count_distributions_respect_domains() {
    let shape = [64i64];
    let mut stream = std::ptr::null_mut();
    let mut output = std::ptr::null_mut();
    // SAFETY: output slots are valid and every successful resource is disposed below.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(19, &mut stream) },
        OK
    );
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_gamma(
                stream,
                shape.as_ptr(),
                1,
                2.0,
                3.0,
                3,
                &mut output,
            )
        },
        OK
    );
    assert!(unsafe { floats(output) }.iter().all(|value| *value >= 0.0));
    unsafe { dispose(output) };
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_standard_normal(
                stream,
                shape.as_ptr(),
                1,
                3,
                &mut output,
            )
        },
        OK
    );
    assert_eq!(unsafe { floats(output) }.len(), 64);
    unsafe { dispose(output) };
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_binomial(
                stream,
                shape.as_ptr(),
                1,
                7,
                0.25,
                2,
                &mut output,
            )
        },
        OK
    );
    assert!(unsafe { integers(output) }
        .iter()
        .all(|value| (0..=7).contains(value)));
    unsafe { dispose(output) };
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_poisson(stream, shape.as_ptr(), 1, 4.0, 2, &mut output)
        },
        OK
    );
    assert!(unsafe { integers(output) }.iter().all(|value| *value >= 0));
    unsafe { dispose(output) };

    output = 1usize as *mut super::super::ffi::TerlanNdArray;
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_beta(
                stream,
                shape.as_ptr(),
                1,
                -1.0,
                2.0,
                3,
                &mut output,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(output.is_null());
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_binomial(
                stream,
                shape.as_ptr(),
                1,
                -1,
                0.5,
                2,
                &mut output,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(output.is_null());
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(stream) },
        OK
    );
}

#[test]
fn all_maintained_distribution_families_execute() {
    let shape = [32i64];
    let mut stream = std::ptr::null_mut();
    let mut output = std::ptr::null_mut();
    // SAFETY: the stream output slot is valid and disposed at the end.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(37, &mut stream) },
        OK
    );

    macro_rules! float_case {
        ($call:expr, $predicate:expr) => {{
            // SAFETY: the stream, shape, and output slot are live for the call.
            assert_eq!(unsafe { $call }, OK);
            // SAFETY: the successful call returned one live Float64 array.
            let values = unsafe { floats(output) };
            assert_eq!(values.len(), 32);
            assert!(values.iter().all($predicate));
            // SAFETY: the test owns the returned array exactly once.
            unsafe { dispose(output) };
        }};
    }
    macro_rules! integer_case {
        ($call:expr, $predicate:expr) => {{
            // SAFETY: the stream, shape, and output slot are live for the call.
            assert_eq!(unsafe { $call }, OK);
            // SAFETY: the successful call returned one live integer array.
            let values = unsafe { integers(output) };
            assert_eq!(values.len(), 32);
            assert!(values.iter().all($predicate));
            // SAFETY: the test owns the returned array exactly once.
            unsafe { dispose(output) };
        }};
    }

    float_case!(
        terlan_ndarray_v1_random_stream_chisquare(stream, shape.as_ptr(), 1, 3.0, 3, &mut output,),
        |value: &f64| *value >= 0.0
    );
    float_case!(
        terlan_ndarray_v1_random_stream_f(stream, shape.as_ptr(), 1, 4.0, 6.0, 3, &mut output,),
        |value: &f64| *value >= 0.0
    );
    float_case!(
        terlan_ndarray_v1_random_stream_gumbel(stream, shape.as_ptr(), 1, 1.0, 2.0, 3, &mut output,),
        |value: &f64| value.is_finite()
    );
    float_case!(
        terlan_ndarray_v1_random_stream_lognormal(
            stream,
            shape.as_ptr(),
            1,
            0.0,
            1.0,
            3,
            &mut output,
        ),
        |value: &f64| *value > 0.0
    );
    float_case!(
        terlan_ndarray_v1_random_stream_pareto(stream, shape.as_ptr(), 1, 3.0, 3, &mut output,),
        |value: &f64| *value >= 0.0
    );
    float_case!(
        terlan_ndarray_v1_random_stream_exponential(stream, shape.as_ptr(), 1, 2.0, 3, &mut output,),
        |value: &f64| *value >= 0.0
    );
    float_case!(
        terlan_ndarray_v1_random_stream_standard_exponential(
            stream,
            shape.as_ptr(),
            1,
            3,
            &mut output,
        ),
        |value: &f64| *value >= 0.0
    );
    float_case!(
        terlan_ndarray_v1_random_stream_standard_gamma(
            stream,
            shape.as_ptr(),
            1,
            2.0,
            3,
            &mut output,
        ),
        |value: &f64| *value >= 0.0
    );
    float_case!(
        terlan_ndarray_v1_random_stream_standard_t(stream, shape.as_ptr(), 1, 5.0, 3, &mut output,),
        |value: &f64| value.is_finite()
    );
    float_case!(
        terlan_ndarray_v1_random_stream_triangular(
            stream,
            shape.as_ptr(),
            1,
            1.0,
            2.0,
            4.0,
            3,
            &mut output,
        ),
        |value: &f64| (1.0..=4.0).contains(value)
    );
    float_case!(
        terlan_ndarray_v1_random_stream_weibull(stream, shape.as_ptr(), 1, 2.0, 3, &mut output,),
        |value: &f64| *value >= 0.0
    );
    float_case!(
        terlan_ndarray_v1_random_stream_standard_cauchy(stream, shape.as_ptr(), 1, 3, &mut output,),
        |value: &f64| value.is_finite()
    );
    integer_case!(
        terlan_ndarray_v1_random_stream_geometric(stream, shape.as_ptr(), 1, 0.4, 2, &mut output,),
        |value: &i64| *value >= 1
    );
    integer_case!(
        terlan_ndarray_v1_random_stream_hypergeometric(
            stream,
            shape.as_ptr(),
            1,
            20,
            10,
            5,
            2,
            &mut output,
        ),
        |value: &i64| (0..=5).contains(value)
    );
    integer_case!(
        terlan_ndarray_v1_random_stream_zipf(stream, shape.as_ptr(), 1, 2.5, 2, &mut output,),
        |value: &i64| *value >= 1
    );

    output = 1usize as *mut super::super::ffi::TerlanNdArray;
    // SAFETY: invalid geometric probability must be rejected without sampling.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_geometric(
                stream,
                shape.as_ptr(),
                1,
                0.0,
                2,
                &mut output,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(output.is_null());
    // SAFETY: the stream remains live after a rejected distribution.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(stream) },
        OK
    );
}

#[test]
fn composed_scalar_distributions_execute_and_reject_invalid_domains() {
    let shape = [64i64];
    let mut stream = std::ptr::null_mut();
    let mut output = std::ptr::null_mut();
    // SAFETY: the stream output slot is valid and all resources are released below.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(113, &mut stream) },
        OK
    );

    macro_rules! float_case {
        ($call:expr, $predicate:expr) => {{
            // SAFETY: the stream, shape, and output slot remain live for the call.
            assert_eq!(unsafe { $call }, OK);
            // SAFETY: a successful call returns one live floating array.
            let values = unsafe { floats(output) };
            assert_eq!(values.len(), 64);
            assert!(values.iter().all($predicate));
            // SAFETY: the test owns the returned array exactly once.
            unsafe { dispose(output) };
        }};
    }

    float_case!(
        terlan_ndarray_v1_random_stream_laplace(
            stream,
            shape.as_ptr(),
            1,
            2.0,
            0.5,
            3,
            &mut output,
        ),
        |value: &f64| value.is_finite()
    );
    float_case!(
        terlan_ndarray_v1_random_stream_logistic(
            stream,
            shape.as_ptr(),
            1,
            -1.0,
            2.0,
            3,
            &mut output,
        ),
        |value: &f64| value.is_finite()
    );
    // SAFETY: the stream, shape, and output slot remain live for the call.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_negative_binomial(
                stream,
                shape.as_ptr(),
                1,
                4.0,
                0.6,
                2,
                &mut output,
            )
        },
        OK
    );
    // SAFETY: the successful call returns one live integer array.
    assert!(unsafe { integers(output) }.iter().all(|value| *value >= 0));
    // SAFETY: the test owns the returned array exactly once.
    unsafe { dispose(output) };
    float_case!(
        terlan_ndarray_v1_random_stream_noncentral_chisquare(
            stream,
            shape.as_ptr(),
            1,
            5.0,
            2.0,
            3,
            &mut output,
        ),
        |value: &f64| *value >= 0.0
    );
    float_case!(
        terlan_ndarray_v1_random_stream_noncentral_f(
            stream,
            shape.as_ptr(),
            1,
            5.0,
            8.0,
            2.0,
            3,
            &mut output,
        ),
        |value: &f64| *value >= 0.0
    );
    float_case!(
        terlan_ndarray_v1_random_stream_rayleigh(stream, shape.as_ptr(), 1, 2.0, 3, &mut output,),
        |value: &f64| *value >= 0.0
    );
    float_case!(
        terlan_ndarray_v1_random_stream_wald(stream, shape.as_ptr(), 1, 2.0, 3.0, 3, &mut output,),
        |value: &f64| *value > 0.0
    );

    output = 1usize as *mut super::super::ffi::TerlanNdArray;
    // SAFETY: invalid domains must clear the output without advancing execution.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_laplace(
                stream,
                shape.as_ptr(),
                1,
                0.0,
                0.0,
                3,
                &mut output,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(output.is_null());
    output = 1usize as *mut super::super::ffi::TerlanNdArray;
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_negative_binomial(
                stream,
                shape.as_ptr(),
                1,
                2.0,
                0.0,
                2,
                &mut output,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(output.is_null());
    output = 1usize as *mut super::super::ffi::TerlanNdArray;
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_noncentral_chisquare(
                stream,
                shape.as_ptr(),
                1,
                0.0,
                1.0,
                3,
                &mut output,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(output.is_null());
    // SAFETY: the stream remains valid following rejected calls.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(stream) },
        OK
    );
}
