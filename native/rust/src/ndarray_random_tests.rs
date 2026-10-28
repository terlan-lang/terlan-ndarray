use super::*;

unsafe fn read_floats(array: *mut super::super::ffi::TerlanNdArray) -> Vec<f64> {
    let mut values = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the test owns a live Float64 array and releases the returned copy.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_to_floats(array, &mut values, &mut length) },
        OK
    );
    if length == 0 {
        return Vec::new();
    }
    // SAFETY: successful readback returns `length` initialized values.
    let result = unsafe { std::slice::from_raw_parts(values, length) }.to_vec();
    // SAFETY: the returned values use the matching package destructor.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_float_values(values) };
    result
}

unsafe fn read_ints(array: *mut super::super::ffi::TerlanNdArray) -> Vec<i64> {
    let mut values = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the test owns a live Int64-compatible array and releases the copy.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_ints_cast(array, &mut values, &mut length)
        },
        OK
    );
    if length == 0 {
        return Vec::new();
    }
    // SAFETY: successful readback returns `length` initialized values.
    let result = unsafe { std::slice::from_raw_parts(values, length) }.to_vec();
    // SAFETY: the returned values use the matching package destructor.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_int_values(values) };
    result
}

unsafe fn read_bools(array: *mut super::super::ffi::TerlanNdArray) -> Vec<bool> {
    let mut values = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the test owns a live Bool array and releases the returned copy.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_to_bools(array, &mut values, &mut length) },
        OK
    );
    if length == 0 {
        return Vec::new();
    }
    // SAFETY: successful readback returns `length` initialized one-byte booleans.
    let result = unsafe { std::slice::from_raw_parts(values, length) }
        .iter()
        .map(|value| *value != 0)
        .collect();
    // SAFETY: the returned values use the matching package destructor.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_bool_values(values) };
    result
}

#[test]
fn random_generators_are_reproducible_and_bounded() {
    let shape = [32i64];
    let mut first = std::ptr::null_mut();
    let mut second = std::ptr::null_mut();
    // SAFETY: arguments are valid and output slots are independently owned.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_uniform(shape.as_ptr(), 1, -2.0, 3.0, 17, 3, &mut first)
        },
        OK
    );
    // SAFETY: arguments are valid and output slots are independently owned.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_uniform(shape.as_ptr(), 1, -2.0, 3.0, 17, 3, &mut second)
        },
        OK
    );
    // SAFETY: both returned arrays are live Float64 values.
    let first_values = unsafe { read_floats(first) };
    // SAFETY: both returned arrays are live Float64 values.
    let second_values = unsafe { read_floats(second) };
    assert_eq!(first_values, second_values);
    assert!(first_values.iter().all(|value| (-2.0..3.0).contains(value)));
    // SAFETY: each successful constructor transfers one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(first) },
        OK
    );
    // SAFETY: each successful constructor transfers one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(second) },
        OK
    );
}

#[test]
fn normal_and_invalid_inputs_have_stable_results() {
    let shape = [4i64];
    let mut array = std::ptr::null_mut();
    // SAFETY: arguments are valid and the output is released below.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_normal(shape.as_ptr(), 1, 4.5, 0.0, 9, 3, &mut array) },
        OK
    );
    // SAFETY: the returned array is a live Float64 value.
    assert_eq!(unsafe { read_floats(array) }, vec![4.5; 4]);
    // SAFETY: the successful constructor transferred one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(array) },
        OK
    );
    // SAFETY: NumPy defines equal uniform bounds as a constant distribution.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_uniform(shape.as_ptr(), 1, 1.0, 1.0, 0, 3, &mut array) },
        OK
    );
    // SAFETY: the returned array is a live Float64 value.
    assert_eq!(unsafe { read_floats(array) }, vec![1.0; 4]);
    // SAFETY: the successful constructor transferred one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(array) },
        OK
    );
    // SAFETY: reversed bounds are rejected before construction.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_uniform(shape.as_ptr(), 1, 2.0, 1.0, 0, 3, &mut array) },
        INVALID_ARGUMENT
    );
    // SAFETY: unsupported dtypes are rejected before construction.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_normal(shape.as_ptr(), 1, 0.0, 1.0, 0, 2, &mut array) },
        UNSUPPORTED_DTYPE
    );
}

#[test]
fn random_boundaries_fail_without_partial_outputs() {
    let empty_shape = [3i64, 0, 2];
    let overflow_shape = [i64::MAX, 2];
    let allocation_shape = [i64::MAX];
    let sample_shape = [64i64];
    let mut array = std::ptr::null_mut();
    // SAFETY: zero-sized shape dimensions are valid borrowed input.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_uniform(empty_shape.as_ptr(), 3, 0.0, 1.0, 1, 3, &mut array)
        },
        OK
    );
    // SAFETY: a live zero-sized Float64 array has an empty readback.
    assert!(unsafe { read_floats(array) }.is_empty());
    // SAFETY: the successful constructor transferred one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(array) },
        OK
    );
    array = std::ptr::null_mut();

    // SAFETY: checked shape multiplication rejects overflow before allocation.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_normal(overflow_shape.as_ptr(), 2, 0.0, 1.0, 1, 3, &mut array)
        },
        SIZE_OVERFLOW
    );
    assert!(array.is_null());
    // SAFETY: impossible reservation is translated to a typed allocation error.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_integers(allocation_shape.as_ptr(), 1, 0, 2, 1, 2, &mut array)
        },
        ALLOCATION_FAILED
    );
    assert!(array.is_null());
    // SAFETY: non-finite distribution bounds are rejected without sampling.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_uniform(
                sample_shape.as_ptr(),
                1,
                f64::NAN,
                1.0,
                1,
                3,
                &mut array,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(array.is_null());

    // SAFETY: endpoint probabilities are valid and produce deterministic constants.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_bernoulli(sample_shape.as_ptr(), 1, 0.0, 1, &mut array) },
        OK
    );
    // SAFETY: the returned Bool array is live.
    assert!(unsafe { read_bools(array) }.iter().all(|value| !*value));
    // SAFETY: the successful constructor transferred one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(array) },
        OK
    );
    array = std::ptr::null_mut();

    // SAFETY: maintained rand rejection sampling handles the extreme signed range.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_integers(
                sample_shape.as_ptr(),
                1,
                i64::MIN,
                i64::MAX,
                2,
                2,
                &mut array,
            )
        },
        OK
    );
    // SAFETY: the live Int64 result has exactly the requested element count.
    assert_eq!(unsafe { read_ints(array) }.len(), 64);
    // SAFETY: the successful constructor transferred one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(array) },
        OK
    );
    // SAFETY: invalid probability and null output are rejected without dereference.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_bernoulli(
                sample_shape.as_ptr(),
                1,
                1.5,
                1,
                std::ptr::null_mut(),
            )
        },
        INVALID_ARGUMENT
    );
}

#[test]
fn integer_bernoulli_and_permutation_are_reproducible() {
    let shape = [64i64];
    let mut integers = std::ptr::null_mut();
    let mut booleans = std::ptr::null_mut();
    let mut first = std::ptr::null_mut();
    let mut second = std::ptr::null_mut();
    // SAFETY: each call receives valid borrowed inputs and an owned output slot.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_integers(shape.as_ptr(), 1, -4, 7, 23, 2, &mut integers)
        },
        OK
    );
    // SAFETY: the returned array is live and integer-compatible.
    let integer_values = unsafe { read_ints(integers) };
    assert!(integer_values.iter().all(|value| (-4..7).contains(value)));
    // SAFETY: each call receives valid borrowed inputs and an owned output slot.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_bernoulli(shape.as_ptr(), 1, 0.25, 23, &mut booleans) },
        OK
    );
    // SAFETY: the returned array is a live Bool array.
    let bool_values = unsafe { read_bools(booleans) };
    assert!(bool_values.iter().any(|value| *value));
    assert!(bool_values.iter().any(|value| !*value));
    // SAFETY: each permutation call has a valid output slot.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_permutation(32, 91, &mut first) },
        OK
    );
    // SAFETY: each permutation call has a valid output slot.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_permutation(32, 91, &mut second) },
        OK
    );
    // SAFETY: both returned arrays are live Int64 arrays.
    assert_eq!(unsafe { read_ints(first) }, unsafe { read_ints(second) });
    // SAFETY: every successful constructor transfers one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(integers) },
        OK
    );
    // SAFETY: every successful constructor transfers one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(booleans) },
        OK
    );
    // SAFETY: every successful constructor transfers one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(first) },
        OK
    );
    // SAFETY: every successful constructor transfers one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(second) },
        OK
    );
}

#[test]
fn functional_shuffle_preserves_values_and_rejects_bad_domains() {
    let shape = [2i64, 4];
    let values = [0i64, 1, 2, 3, 4, 5, 6, 7];
    let mut source = std::ptr::null_mut();
    let mut first = std::ptr::null_mut();
    let mut second = std::ptr::null_mut();
    // SAFETY: source slices remain live for the copying constructor.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_from_ints(
                values.as_ptr(),
                values.len() as i64,
                shape.as_ptr(),
                shape.len() as i64,
                &mut source,
            )
        },
        OK
    );
    // SAFETY: source is live and each output slot is independently owned.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_shuffle(source, 1, 19, &mut first) },
        OK
    );
    // SAFETY: source is live and each output slot is independently owned.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_shuffle(source, -1, 19, &mut second) },
        OK
    );
    // SAFETY: all returned arrays are live integer-compatible arrays.
    assert_eq!(unsafe { read_ints(first) }, unsafe { read_ints(second) });
    // SAFETY: the source remains readable after functional shuffle.
    assert_eq!(unsafe { read_ints(source) }, values);
    // SAFETY: the second successful shuffle transferred one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(second) },
        OK
    );
    second = std::ptr::null_mut();
    // SAFETY: invalid axes are rejected before constructing an output.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_shuffle(source, 2, 0, &mut second) },
        INVALID_ARGUMENT
    );
    assert!(second.is_null());
    // SAFETY: malformed domains are rejected before reading shape memory.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_integers(shape.as_ptr(), 2, 4, 4, 0, 2, &mut second) },
        INVALID_ARGUMENT
    );
    // SAFETY: every successful constructor transfers one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(first) },
        OK
    );
    // SAFETY: every successful constructor transfers one owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(source) },
        OK
    );
}

#[test]
fn owned_stream_advances_reproducibly_across_all_operations() {
    let shape = [2i64, 4];
    let values = [0i64, 1, 2, 3, 4, 5, 6, 7];
    let mut first_stream = std::ptr::null_mut();
    let mut second_stream = std::ptr::null_mut();
    // SAFETY: both output slots receive independently owned streams.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(73, &mut first_stream) },
        OK
    );
    // SAFETY: both output slots receive independently owned streams.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(73, &mut second_stream) },
        OK
    );

    let mut first_uniform = std::ptr::null_mut();
    let mut second_uniform = std::ptr::null_mut();
    let mut advanced_uniform = std::ptr::null_mut();
    // SAFETY: streams and borrowed shapes are live for every call.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_uniform(
                first_stream,
                shape.as_ptr(),
                2,
                -1.0,
                1.0,
                3,
                &mut first_uniform,
            )
        },
        OK
    );
    // SAFETY: streams and borrowed shapes are live for every call.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_uniform(
                second_stream,
                shape.as_ptr(),
                2,
                -1.0,
                1.0,
                3,
                &mut second_uniform,
            )
        },
        OK
    );
    // SAFETY: the first stream remains live and advances again.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_uniform(
                first_stream,
                shape.as_ptr(),
                2,
                -1.0,
                1.0,
                3,
                &mut advanced_uniform,
            )
        },
        OK
    );
    // SAFETY: all three arrays are live Float64 arrays.
    let first_uniform_values = unsafe { read_floats(first_uniform) };
    // SAFETY: all three arrays are live Float64 arrays.
    assert_eq!(first_uniform_values, unsafe { read_floats(second_uniform) });
    assert_eq!(
        first_uniform_values,
        vec![
            -0.9511954521741668,
            0.06863333301204522,
            0.03697361429363122,
            0.5409061969874509,
            0.49454083998355136,
            0.6207889389170667,
            -0.6941178387537341,
            -0.3719899325320102,
        ]
    );
    // SAFETY: all three arrays are live Float64 arrays.
    assert_ne!(first_uniform_values, unsafe {
        read_floats(advanced_uniform)
    });

    let mut normal = std::ptr::null_mut();
    let mut integers = std::ptr::null_mut();
    let mut booleans = std::ptr::null_mut();
    let mut permutation = std::ptr::null_mut();
    // SAFETY: each operation borrows the stream and transfers one array.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_normal(
                first_stream,
                shape.as_ptr(),
                2,
                0.0,
                1.0,
                3,
                &mut normal,
            )
        },
        OK
    );
    // SAFETY: each operation borrows the stream and transfers one array.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_integers(
                first_stream,
                shape.as_ptr(),
                2,
                -5,
                5,
                2,
                &mut integers,
            )
        },
        OK
    );
    // SAFETY: each operation borrows the stream and transfers one array.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_bernoulli(
                first_stream,
                shape.as_ptr(),
                2,
                0.5,
                &mut booleans,
            )
        },
        OK
    );
    // SAFETY: each operation borrows the stream and transfers one array.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_permutation(first_stream, 8, &mut permutation) },
        OK
    );

    let mut source = std::ptr::null_mut();
    let mut shuffled = std::ptr::null_mut();
    // SAFETY: source slices remain live for the copying constructor.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_from_ints(
                values.as_ptr(),
                8,
                shape.as_ptr(),
                2,
                &mut source,
            )
        },
        OK
    );
    // SAFETY: stream and source are live and the output is owned.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_shuffle(first_stream, source, 1, &mut shuffled) },
        OK
    );
    // SAFETY: all operation results are live arrays with compatible readback types.
    assert_eq!(
        unsafe { read_floats(normal) },
        vec![
            -1.3012895064264287,
            0.007428118903149798,
            0.923279163165256,
            -0.2433335392644783,
            0.6589691530322335,
            -0.09838380785274985,
            1.2471744216587757,
            1.5311346615938863,
        ]
    );
    // SAFETY: all operation results are live arrays with compatible readback types.
    assert_eq!(unsafe { read_ints(integers) }, [2, 2, 0, 4, 0, -3, 2, -2]);
    // SAFETY: all operation results are live arrays with compatible readback types.
    assert_eq!(
        unsafe { read_bools(booleans) },
        [true, true, true, false, true, true, true, false]
    );
    // SAFETY: all operation results are live arrays with compatible readback types.
    assert_eq!(unsafe { read_ints(permutation) }, [2, 7, 6, 1, 0, 3, 4, 5]);
    // SAFETY: all operation results are live arrays with compatible readback types.
    assert_eq!(unsafe { read_ints(shuffled) }, [1, 2, 0, 3, 5, 6, 4, 7]);

    for array in [
        first_uniform,
        second_uniform,
        advanced_uniform,
        normal,
        integers,
        booleans,
        permutation,
        source,
        shuffled,
    ] {
        // SAFETY: every array in the list is a distinct owned result.
        assert_eq!(
            unsafe { super::super::ffi::terlan_ndarray_v1_delete(array) },
            OK
        );
    }
    // SAFETY: each stream is transferred to its destructor exactly once.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(first_stream) },
        OK
    );
    // SAFETY: each stream is transferred to its destructor exactly once.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(second_stream) },
        OK
    );
    // SAFETY: null handles are rejected without dereference.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(std::ptr::null_mut()) },
        INVALID_ARGUMENT
    );
}
