use super::*;

type Array = *mut super::super::ffi::TerlanNdArray;
type Stream = *mut super::super::ffi::TerlanNdRandomStream;

/// Creates one deterministic package-owned random stream.
fn stream(seed: i64) -> Stream {
    let mut output = std::ptr::null_mut();
    // SAFETY: the output slot is valid and receives one owned stream.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(seed, &mut output) },
        OK
    );
    output
}

/// Reads numerical array values as owned Float64 test data.
fn values(array: Array) -> Vec<f64> {
    let mut output = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the array is live and both output slots are valid.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_floats_cast(array, &mut output, &mut length)
        },
        OK
    );
    if length == 0 {
        return Vec::new();
    }
    // SAFETY: successful readback returns `length` initialized values.
    let values = unsafe { std::slice::from_raw_parts(output, length) }.to_vec();
    // SAFETY: copied values are released through their matching destructor.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_float_values(output) };
    values
}

/// Reads an array's shape as owned test data.
fn shape(array: Array) -> Vec<i64> {
    let mut rank = 0i64;
    let mut output = std::ptr::null_mut();
    // SAFETY: the array is live and both output slots are valid.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_rank(array, &mut rank) },
        OK
    );
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_shape(array, &mut output) },
        OK
    );
    if rank == 0 {
        return Vec::new();
    }
    // SAFETY: successful shape readback exposes `rank` borrowed dimensions.
    unsafe { std::slice::from_raw_parts(output, rank as usize) }.to_vec()
}

/// Reads one array's ABI dtype code.
fn dtype(array: Array) -> i32 {
    let mut output = -1;
    // SAFETY: the array is live and the scalar output slot is valid.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_dtype(array, &mut output) },
        OK
    );
    output
}

/// Releases one package-owned array.
fn dispose_array(array: Array) {
    // SAFETY: each successful call transfers exactly one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(array) },
        OK
    );
}

/// Releases one package-owned random stream.
fn dispose_stream(stream: Stream) {
    // SAFETY: each stream is released exactly once after its final use.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(stream) },
        OK
    );
}

#[test]
fn power_distribution_is_reproducible_bounded_shaped_and_typed() {
    let dimensions = [2i64, 32];
    let first_stream = stream(919);
    let second_stream = stream(919);
    let mut first = std::ptr::null_mut();
    let mut second = std::ptr::null_mut();

    // SAFETY: streams, shape, and output slots satisfy the generated ABI.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_power(
                first_stream,
                dimensions.as_ptr(),
                2,
                3.5,
                14,
                &mut first,
            )
        },
        OK
    );
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_power(
                second_stream,
                dimensions.as_ptr(),
                2,
                3.5,
                14,
                &mut second,
            )
        },
        OK
    );

    let first_values = values(first);
    assert_eq!(first_values, values(second));
    assert!(first_values.iter().all(|value| (0.0..=1.0).contains(value)));
    assert_eq!(shape(first), dimensions);
    assert_eq!(dtype(first), 14);

    dispose_array(second);
    dispose_array(first);
    dispose_stream(second_stream);
    dispose_stream(first_stream);
}

#[test]
fn power_distribution_handles_scalar_and_zero_extent_shapes() {
    let stream = stream(920);
    let zero_extent = [2i64, 0, 3];
    let mut scalar = std::ptr::null_mut();
    let mut empty = std::ptr::null_mut();

    // SAFETY: a null shape is valid for rank zero and the output slot is valid.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_power(stream, std::ptr::null(), 0, 1.0, 3, &mut scalar)
        },
        OK
    );
    assert_eq!(shape(scalar), Vec::<i64>::new());
    assert_eq!(values(scalar).len(), 1);

    // SAFETY: the zero-extent shape and output slot satisfy the generated ABI.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_power(
                stream,
                zero_extent.as_ptr(),
                3,
                2.0,
                11,
                &mut empty,
            )
        },
        OK
    );
    assert_eq!(shape(empty), zero_extent);
    assert!(values(empty).is_empty());
    assert_eq!(dtype(empty), 11);

    dispose_array(empty);
    dispose_array(scalar);
    dispose_stream(stream);
}

#[test]
fn power_distribution_rejects_invalid_domains_without_advancing_stream() {
    let dimensions = [16i64];
    let tested = stream(921);
    let control = stream(921);
    let mut rejected = 1usize as Array;

    for distribution_shape in [0.0, -1.0, f64::NEG_INFINITY] {
        // SAFETY: invalid scalar domains must be rejected before sampling.
        assert_eq!(
            unsafe {
                terlan_ndarray_v1_random_stream_power(
                    tested,
                    dimensions.as_ptr(),
                    1,
                    distribution_shape,
                    3,
                    &mut rejected,
                )
            },
            INVALID_ARGUMENT
        );
        assert!(rejected.is_null());
        rejected = 1usize as Array;
    }

    let mut actual = std::ptr::null_mut();
    let mut expected = std::ptr::null_mut();
    // SAFETY: both streams and output slots are valid for the accepted calls.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_power(
                tested,
                dimensions.as_ptr(),
                1,
                2.0,
                3,
                &mut actual,
            )
        },
        OK
    );
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_power(
                control,
                dimensions.as_ptr(),
                1,
                2.0,
                3,
                &mut expected,
            )
        },
        OK
    );
    assert_eq!(values(actual), values(expected));

    dispose_array(expected);
    dispose_array(actual);
    dispose_stream(control);
    dispose_stream(tested);
}

#[test]
fn power_distribution_matches_numpy_nan_and_infinite_parameter_semantics() {
    let dimensions = [8i64];
    let stream = stream(922);
    let mut nan_values = std::ptr::null_mut();
    let mut infinite_values = std::ptr::null_mut();

    // SAFETY: the stream, shape, and output slots satisfy the generated ABI.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_power(
                stream,
                dimensions.as_ptr(),
                1,
                f64::NAN,
                3,
                &mut nan_values,
            )
        },
        OK
    );
    assert!(values(nan_values).iter().all(|value| value.is_nan()));

    // SAFETY: positive infinity is a valid NumPy power parameter.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_power(
                stream,
                dimensions.as_ptr(),
                1,
                f64::INFINITY,
                3,
                &mut infinite_values,
            )
        },
        OK
    );
    assert_eq!(values(infinite_values), vec![1.0; 8]);

    dispose_array(infinite_values);
    dispose_array(nan_values);
    dispose_stream(stream);
}
