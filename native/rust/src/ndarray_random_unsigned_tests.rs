use super::*;

type Array = *mut super::super::ffi::TerlanNdArray;
type Stream = *mut super::super::ffi::TerlanNdRandomStream;

/// Creates one exact UInt64 array without signed or floating conversion.
fn unsigned_array(values: &[u64], shape: &[i64]) -> Array {
    let mut output = std::ptr::null_mut();
    // SAFETY: both borrowed slices remain live while the constructor copies them.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_from_uints(
                values.as_ptr(),
                values.len() as i64,
                shape.as_ptr(),
                shape.len() as i64,
                &mut output,
            )
        },
        OK
    );
    output
}

/// Creates one exact signed integer array with an explicit dtype.
fn signed_array(values: &[i64], shape: &[i64], dtype: i32) -> Array {
    let mut output = std::ptr::null_mut();
    // SAFETY: both borrowed slices remain live while the constructor copies them.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_from_ints_as(
                values.as_ptr(),
                values.len() as i64,
                shape.as_ptr(),
                shape.len() as i64,
                dtype,
                &mut output,
            )
        },
        OK
    );
    output
}

/// Reads exact nonnegative values through the UInt64 cast path.
fn unsigned_values(array: Array) -> Vec<u64> {
    let mut output = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the live array and writable output slots satisfy the ABI.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_uints_cast(array, &mut output, &mut length)
        },
        OK
    );
    let values = if length == 0 {
        Vec::new()
    } else {
        // SAFETY: successful nonempty readback exposes `length` initialized values.
        unsafe { std::slice::from_raw_parts(output, length) }.to_vec()
    };
    // SAFETY: the copied values are released through their matching destructor.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_uint_values(output) };
    values
}

/// Creates one deterministic package-owned random stream.
fn stream(seed: i64) -> Stream {
    let mut output = std::ptr::null_mut();
    // SAFETY: the writable output slot receives one owned stream.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(seed, &mut output) },
        OK
    );
    output
}

/// Samples one exact integer-bound family and returns its owned output.
fn sample(stream: Stream, low: Array, high: Array, distribution: i64, dtype: i32) -> Array {
    let handles = [low as usize as u64, high as usize as u64];
    let mut output = std::ptr::null_mut();
    // SAFETY: the stream, handles, and writable output remain live for the call.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_sample_broadcast(
                stream,
                handles.as_ptr(),
                handles.len() as i64,
                distribution,
                dtype,
                &mut output,
            )
        },
        OK
    );
    output
}

/// Releases one package-owned array.
fn dispose_array(array: Array) {
    // SAFETY: each test releases each owned array exactly once.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(array) },
        OK
    );
}

/// Releases one package-owned random stream.
fn dispose_stream(stream: Stream) {
    // SAFETY: each test releases each owned stream exactly once.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(stream) },
        OK
    );
}

#[test]
fn uint64_abi_round_trips_the_full_storage_domain() {
    let expected = [0, i64::MAX as u64, i64::MAX as u64 + 1, u64::MAX];
    let array = unsigned_array(&expected, &[2, 2]);
    assert_eq!(unsigned_values(array), expected);

    let negative = signed_array(&[-1], &[], 2);
    let mut values = 1usize as *mut u64;
    let mut length = usize::MAX;
    // SAFETY: a live negative array is intentionally passed to the checked cast.
    assert_ne!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_uints_cast(negative, &mut values, &mut length)
        },
        OK
    );
    assert!(values.is_null());
    assert_eq!(length, 0);

    let fractional = {
        let mut output = std::ptr::null_mut();
        let value = [1.5];
        // SAFETY: the scalar input and output slot remain live for the copying call.
        assert_eq!(
            unsafe {
                super::super::ffi::terlan_ndarray_v1_from_floats_as(
                    value.as_ptr(),
                    1,
                    std::ptr::null(),
                    0,
                    3,
                    &mut output,
                )
            },
            OK
        );
        output
    };
    values = 1usize as *mut u64;
    length = usize::MAX;
    // SAFETY: a live fractional array is intentionally passed to the checked cast.
    assert_ne!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_uints_cast(fractional, &mut values, &mut length)
        },
        OK
    );
    assert!(values.is_null());
    assert_eq!(length, 0);

    // SAFETY: malformed pointer contracts must fail before dereference.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_uints_cast(
                std::ptr::null_mut(),
                &mut values,
                &mut length,
            )
        },
        INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_uints_cast(
                array,
                std::ptr::null_mut(),
                &mut length,
            )
        },
        INVALID_ARGUMENT
    );

    dispose_array(fractional);
    dispose_array(negative);
    dispose_array(array);
}

#[test]
fn closed_uint64_sampling_reaches_exact_singleton_and_full_range_bounds() {
    let stream = stream(1301);
    let maximum = unsigned_array(&[u64::MAX], &[]);
    let singleton = sample(stream, maximum, maximum, 30, 10);
    assert_eq!(unsigned_values(singleton), vec![u64::MAX]);

    let low = unsigned_array(&[0; 256], &[256]);
    let high = unsigned_array(&[u64::MAX; 256], &[256]);
    let full_range = sample(stream, low, high, 30, 10);
    let values = unsigned_values(full_range);
    assert_eq!(values.len(), 256);
    assert!(values.iter().any(|value| *value > i64::MAX as u64));

    for array in [full_range, high, low, singleton, maximum] {
        dispose_array(array);
    }
    dispose_stream(stream);
}

#[test]
fn mixed_signed_and_unsigned_bounds_cross_the_int64_boundary_exactly() {
    let stream = stream(1302);
    let low = signed_array(&[i64::MAX], &[], 2);
    let high = unsigned_array(&[i64::MAX as u64 + 1], &[]);
    let unsigned = sample(stream, low, high, 29, 10);
    assert_eq!(unsigned_values(unsigned), vec![i64::MAX as u64]);

    let signed = sample(stream, low, high, 29, 2);
    let mut signed_values = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the output is Int64 and the copied values are released below.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_ints_cast(
                signed,
                &mut signed_values,
                &mut length,
            )
        },
        OK
    );
    assert_eq!(
        unsafe { std::slice::from_raw_parts(signed_values, length) },
        &[i64::MAX]
    );
    // SAFETY: the copied values use the matching destructor.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_int_values(signed_values) };

    let high_closed = unsigned_array(&[i64::MAX as u64], &[]);
    let closed = sample(stream, low, high_closed, 30, 2);
    let mut closed_values = std::ptr::null_mut();
    let mut closed_length = 0usize;
    // SAFETY: the closed singleton output is Int64 and is released below.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_ints_cast(
                closed,
                &mut closed_values,
                &mut closed_length,
            )
        },
        OK
    );
    assert_eq!(
        unsafe { std::slice::from_raw_parts(closed_values, closed_length) },
        &[i64::MAX]
    );
    // SAFETY: the copied values use the matching destructor.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_int_values(closed_values) };

    for array in [closed, high_closed, signed, unsigned, high, low] {
        dispose_array(array);
    }
    dispose_stream(stream);
}

#[test]
fn broadcast_uint64_endpoint_pairs_preserve_each_exact_interval() {
    let stream = stream(1303);
    let low = unsigned_array(&[i64::MAX as u64, i64::MAX as u64 + 1], &[2]);
    let high = unsigned_array(&[i64::MAX as u64 + 1, i64::MAX as u64 + 2], &[2]);
    let output = sample(stream, low, high, 29, 10);
    assert_eq!(
        unsigned_values(output),
        vec![i64::MAX as u64, i64::MAX as u64 + 1]
    );

    for array in [output, high, low] {
        dispose_array(array);
    }
    dispose_stream(stream);
}

#[test]
fn invalid_uint64_output_bounds_clear_output_and_preserve_rng_state() {
    let tested = stream(1304);
    let control = stream(1304);
    let negative = signed_array(&[-1], &[], 2);
    let one = unsigned_array(&[1], &[]);
    let maximum = unsigned_array(&[u64::MAX], &[]);
    let handles = [negative as usize as u64, one as usize as u64];
    let mut rejected = 1usize as Array;

    // SAFETY: live bounds intentionally violate the UInt64 output domain.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_sample_broadcast(
                tested,
                handles.as_ptr(),
                2,
                29,
                10,
                &mut rejected,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(rejected.is_null());

    let too_wide = [one as usize as u64, maximum as usize as u64];
    rejected = 1usize as Array;
    // SAFETY: the closed upper bound exceeds the requested Int64 output domain.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_sample_broadcast(
                tested,
                too_wide.as_ptr(),
                2,
                30,
                2,
                &mut rejected,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(rejected.is_null());

    let actual = sample(tested, one, one, 30, 10);
    let expected = sample(control, one, one, 30, 10);
    assert_eq!(unsigned_values(actual), unsigned_values(expected));

    for array in [expected, actual, maximum, one, negative] {
        dispose_array(array);
    }
    dispose_stream(control);
    dispose_stream(tested);
}

#[test]
fn integer_fuzz_samples_remain_in_bounds_for_every_dtype_and_endpoint() {
    const COUNT: usize = 1 << 16;
    let stream = stream(1305);
    let low = signed_array(&[2; COUNT], &[COUNT as i64], 2);
    for (distribution, high_value) in [(29, 16), (30, 15)] {
        let high = signed_array(&[high_value; COUNT], &[COUNT as i64], 2);
        for dtype in [2, 4, 5, 6, 7, 8, 9, 10] {
            let output = sample(stream, low, high, distribution, dtype);
            let values = unsigned_values(output);
            assert_eq!(values.len(), COUNT);
            assert!(values.iter().all(|value| (2..16).contains(value)));
            dispose_array(output);
        }
        dispose_array(high);
    }

    let bool_low = signed_array(&[0; COUNT], &[COUNT as i64], 2);
    for (distribution, high_value) in [(29, 2), (30, 1)] {
        let bool_high = signed_array(&[high_value; COUNT], &[COUNT as i64], 2);
        let output = sample(stream, bool_low, bool_high, distribution, 1);
        assert!(unsigned_values(output).iter().all(|value| *value < 2));
        dispose_array(output);
        dispose_array(bool_high);
    }

    dispose_array(bool_low);
    dispose_array(low);
    dispose_stream(stream);
}
