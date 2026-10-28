use super::*;

type Array = *mut super::super::ffi::TerlanNdArray;

unsafe fn float_array(values: &[f64], shape: &[i64]) -> Array {
    let mut output = std::ptr::null_mut();
    // SAFETY: both slices remain live while the constructor copies them.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_from_floats_as(
                values.as_ptr(),
                values.len() as i64,
                shape.as_ptr(),
                shape.len() as i64,
                3,
                &mut output,
            )
        },
        OK
    );
    output
}

unsafe fn integer_array(values: &[i64], shape: &[i64]) -> Array {
    let mut output = std::ptr::null_mut();
    // SAFETY: both slices remain live while the constructor copies them.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_from_ints_as(
                values.as_ptr(),
                values.len() as i64,
                shape.as_ptr(),
                shape.len() as i64,
                2,
                &mut output,
            )
        },
        OK
    );
    output
}

unsafe fn array_shape(array: Array) -> Vec<i64> {
    // SAFETY: tests only pass live package arrays.
    unsafe { super::native_support::shape(array).unwrap() }
}

unsafe fn float_values(array: Array) -> Vec<f64> {
    // SAFETY: tests only pass live numeric package arrays.
    unsafe { super::native_support::floats(array, true).unwrap() }
}

unsafe fn integer_values(array: Array) -> Vec<i64> {
    let mut values = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the array is live and the returned allocation is released below.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_ints_cast(array, &mut values, &mut length)
        },
        OK
    );
    // SAFETY: successful readback returns exactly `length` initialized values.
    let output = unsafe { std::slice::from_raw_parts(values, length) }.to_vec();
    // SAFETY: the copied allocation uses its matching package destructor.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_int_values(values) };
    output
}

unsafe fn dispose(array: Array) {
    // SAFETY: each test passes an independently owned live array exactly once.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(array) },
        OK
    );
}

#[test]
fn structured_distributions_preserve_shapes_and_conservation_contracts() {
    let mut stream = std::ptr::null_mut();
    let mut output = std::ptr::null_mut();
    // SAFETY: the stream output slot is valid and released below.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(211, &mut stream) },
        OK
    );
    let source = unsafe { integer_array(&[10, 20, 30, 40], &[4]) };
    let weights = unsafe { float_array(&[0.1, 0.2, 0.3, 0.4], &[4]) };
    let sample_shape = [3i64];
    // SAFETY: all borrowed resources and output storage are live.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_choice(
                stream,
                source,
                sample_shape.as_ptr(),
                1,
                weights,
                0,
                &mut output,
            )
        },
        OK
    );
    assert_eq!(unsafe { array_shape(output) }, [3]);
    let selected = unsafe { integer_values(output) };
    assert!(selected
        .iter()
        .all(|value| [10, 20, 30, 40].contains(value)));
    let mut unique = selected.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), 3);
    unsafe { dispose(output) };

    let alpha = unsafe { float_array(&[1.0, 2.0, 3.0], &[3]) };
    let batch = [8i64];
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_dirichlet(
                stream,
                alpha,
                batch.as_ptr(),
                1,
                3,
                &mut output,
            )
        },
        OK
    );
    assert_eq!(unsafe { array_shape(output) }, [8, 3]);
    for row in unsafe { float_values(output) }.chunks_exact(3) {
        assert!(row.iter().all(|value| *value >= 0.0));
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    }
    unsafe { dispose(output) };

    let probabilities = unsafe { float_array(&[0.2, 0.3, 0.5], &[3]) };
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_multinomial(
                stream,
                10,
                probabilities,
                batch.as_ptr(),
                1,
                2,
                &mut output,
            )
        },
        OK
    );
    assert_eq!(unsafe { array_shape(output) }, [8, 3]);
    for row in unsafe { integer_values(output) }.chunks_exact(3) {
        assert!(row.iter().all(|value| *value >= 0));
        assert_eq!(row.iter().sum::<i64>(), 10);
    }
    unsafe { dispose(output) };

    let colors = unsafe { integer_array(&[8, 6, 4], &[3]) };
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_multivariate_hypergeometric(
                stream,
                colors,
                5,
                batch.as_ptr(),
                1,
                2,
                &mut output,
            )
        },
        OK
    );
    assert_eq!(unsafe { array_shape(output) }, [8, 3]);
    for row in unsafe { integer_values(output) }.chunks_exact(3) {
        assert_eq!(row.iter().sum::<i64>(), 5);
    }
    unsafe { dispose(output) };

    let mean = unsafe { float_array(&[1.0, -2.0], &[2]) };
    let covariance = unsafe { float_array(&[2.0, 0.5, 0.5, 1.0], &[2, 2]) };
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_multivariate_normal(
                stream,
                mean,
                covariance,
                batch.as_ptr(),
                1,
                3,
                &mut output,
            )
        },
        OK
    );
    assert_eq!(unsafe { array_shape(output) }, [8, 2]);
    assert!(unsafe { float_values(output) }
        .iter()
        .all(|value| value.is_finite()));
    unsafe { dispose(output) };

    let scalar_shape = [64i64];
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_logseries(
                stream,
                scalar_shape.as_ptr(),
                1,
                0.7,
                2,
                &mut output,
            )
        },
        OK
    );
    assert!(unsafe { integer_values(output) }
        .iter()
        .all(|value| *value >= 1));
    unsafe { dispose(output) };
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_vonmises(
                stream,
                scalar_shape.as_ptr(),
                1,
                0.0,
                4.0,
                3,
                &mut output,
            )
        },
        OK
    );
    assert!(unsafe { float_values(output) }
        .iter()
        .all(|value| (-std::f64::consts::PI..=std::f64::consts::PI).contains(value)));
    unsafe { dispose(output) };

    unsafe { dispose(source) };
    unsafe { dispose(weights) };
    unsafe { dispose(alpha) };
    unsafe { dispose(probabilities) };
    unsafe { dispose(colors) };
    unsafe { dispose(mean) };
    unsafe { dispose(covariance) };
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(stream) },
        OK
    );
}

#[test]
fn structured_distributions_reject_invalid_probability_and_covariance_domains() {
    let mut stream = std::ptr::null_mut();
    let mut output = 1usize as Array;
    // SAFETY: the stream output slot is valid and released below.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(223, &mut stream) },
        OK
    );
    let bad_probabilities = unsafe { float_array(&[0.2, 0.2], &[2]) };
    let shape = [2i64];
    // SAFETY: invalid probability sum must be rejected without an output.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_multinomial(
                stream,
                4,
                bad_probabilities,
                shape.as_ptr(),
                1,
                2,
                &mut output,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(output.is_null());
    let mean = unsafe { float_array(&[0.0, 0.0], &[2]) };
    let nonsymmetric = unsafe { float_array(&[1.0, 2.0, 0.0, 1.0], &[2, 2]) };
    output = 1usize as Array;
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_multivariate_normal(
                stream,
                mean,
                nonsymmetric,
                shape.as_ptr(),
                1,
                3,
                &mut output,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(output.is_null());
    output = 1usize as Array;
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_logseries(
                stream,
                shape.as_ptr(),
                1,
                1.0,
                2,
                &mut output,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(output.is_null());
    unsafe { dispose(bad_probabilities) };
    unsafe { dispose(mean) };
    unsafe { dispose(nonsymmetric) };
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(stream) },
        OK
    );
}

#[test]
fn choice_replaces_arbitrary_source_axes_with_sample_dimensions() {
    let mut stream = std::ptr::null_mut();
    let mut output = std::ptr::null_mut();
    // SAFETY: the writable stream slot receives one owned stream.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(227, &mut stream) },
        OK
    );
    let source = unsafe { integer_array(&[0, 1, 2, 3, 4, 5, 6, 7], &[4, 2]) };
    let row_probabilities = unsafe { float_array(&[0.1, 0.2, 0.3, 0.4], &[4]) };
    let column_probabilities = unsafe { float_array(&[0.5, 0.5], &[2]) };

    let rows = [3i64];
    // SAFETY: every resource and borrowed shape remains live for the call.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_choice(
                stream,
                source,
                rows.as_ptr(),
                1,
                row_probabilities,
                1,
                &mut output,
            )
        },
        OK
    );
    assert_eq!(unsafe { array_shape(output) }, [3, 2]);
    for row in unsafe { integer_values(output) }.chunks_exact(2) {
        assert!(matches!(row, [0, 1] | [2, 3] | [4, 5] | [6, 7]));
    }
    unsafe { dispose(output) };

    let inserted = [2i64, 2];
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_choice_axis(
                stream,
                source,
                inserted.as_ptr(),
                2,
                column_probabilities,
                1,
                -1,
                1,
                &mut output,
            )
        },
        OK
    );
    assert_eq!(unsafe { array_shape(output) }, [4, 2, 2]);
    for (row_index, row) in unsafe { integer_values(output) }
        .chunks_exact(4)
        .enumerate()
    {
        let first = (row_index * 2) as i64;
        assert!(row
            .iter()
            .all(|value| *value == first || *value == first + 1));
    }
    unsafe { dispose(output) };

    let all_columns = [2i64];
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_choice_axis(
                stream,
                source,
                all_columns.as_ptr(),
                1,
                column_probabilities,
                0,
                1,
                0,
                &mut output,
            )
        },
        OK
    );
    assert_eq!(unsafe { array_shape(output) }, [4, 2]);
    assert_eq!(unsafe { integer_values(output) }, [0, 1, 2, 3, 4, 5, 6, 7]);
    unsafe { dispose(output) };

    let uniform_columns = [3i64];
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_choice_uniform_axis(
                stream,
                source,
                uniform_columns.as_ptr(),
                1,
                1,
                1,
                1,
                &mut output,
            )
        },
        OK
    );
    assert_eq!(unsafe { array_shape(output) }, [4, 3]);
    for (row_index, row) in unsafe { integer_values(output) }
        .chunks_exact(3)
        .enumerate()
    {
        let first = (row_index * 2) as i64;
        assert!(row
            .iter()
            .all(|value| *value == first || *value == first + 1));
    }
    unsafe { dispose(output) };

    unsafe { dispose(source) };
    unsafe { dispose(row_probabilities) };
    unsafe { dispose(column_probabilities) };
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(stream) },
        OK
    );
}

#[test]
fn choice_axis_rejections_preserve_stream_state_and_clear_outputs() {
    let mut tested = std::ptr::null_mut();
    let mut control = std::ptr::null_mut();
    // SAFETY: both writable slots receive independent owned streams.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(229, &mut tested) },
        OK
    );
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(229, &mut control) },
        OK
    );
    let source = unsafe { integer_array(&[0, 1, 2, 3, 4, 5], &[2, 3]) };
    let valid = unsafe { float_array(&[0.2, 0.3, 0.5], &[3]) };
    let wrong_sum = unsafe { float_array(&[0.2, 0.2, 0.2], &[3]) };
    let wrong_shape = unsafe { float_array(&[0.5, 0.5], &[2]) };
    let sample_shape = [2i64];

    for (probabilities, replacement, axis, shuffle, status) in [
        (wrong_sum, 1, 1, 1, INVALID_ARGUMENT),
        (wrong_shape, 1, 1, 1, INVALID_ARGUMENT),
        (valid, 1, 2, 1, super::native_support::AXIS_OUT_OF_RANGE),
        (valid, 2, 1, 1, INVALID_ARGUMENT),
        (valid, 1, 1, 2, INVALID_ARGUMENT),
    ] {
        let mut rejected = 1usize as Array;
        // SAFETY: malformed scalar contracts are intentionally passed with live resources.
        assert_eq!(
            unsafe {
                terlan_ndarray_v1_random_stream_choice_axis(
                    tested,
                    source,
                    sample_shape.as_ptr(),
                    1,
                    probabilities,
                    replacement,
                    axis,
                    shuffle,
                    &mut rejected,
                )
            },
            status
        );
        assert!(rejected.is_null());
    }

    let mut actual = std::ptr::null_mut();
    let mut expected = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_choice_axis(
                tested,
                source,
                sample_shape.as_ptr(),
                1,
                valid,
                1,
                1,
                1,
                &mut actual,
            )
        },
        OK
    );
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_choice_axis(
                control,
                source,
                sample_shape.as_ptr(),
                1,
                valid,
                1,
                1,
                1,
                &mut expected,
            )
        },
        OK
    );
    assert_eq!(unsafe { integer_values(actual) }, unsafe {
        integer_values(expected)
    });

    unsafe { dispose(actual) };
    unsafe { dispose(expected) };
    unsafe { dispose(source) };
    unsafe { dispose(valid) };
    unsafe { dispose(wrong_sum) };
    unsafe { dispose(wrong_shape) };
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(control) },
        OK
    );
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(tested) },
        OK
    );
}
