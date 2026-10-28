use super::*;

type Array = *mut super::super::ffi::TerlanNdArray;
type Stream = *mut super::super::ffi::TerlanNdRandomStream;

/// Creates one owned floating array for a test parameter.
fn array(values: &[f64], shape: &[i64], dtype: i32) -> Array {
    let mut output = std::ptr::null_mut();
    // SAFETY: both borrowed slices remain live for the copying constructor.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_from_floats_as(
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

/// Creates one owned integer array for an exact-bound test parameter.
fn integer_array(values: &[i64], shape: &[i64], dtype: i32) -> Array {
    let mut output = std::ptr::null_mut();
    // SAFETY: both borrowed slices remain live for the copying constructor.
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

/// Encodes borrowed array handles for the package-private C ABI.
fn handles(arrays: &[Array]) -> Vec<u64> {
    arrays.iter().map(|array| *array as usize as u64).collect()
}

/// Samples one generic broadcast family and requires success.
fn sample(stream: Stream, arrays: &[Array], distribution: i64, dtype: i32) -> Array {
    let handles = handles(arrays);
    let mut output = std::ptr::null_mut();
    // SAFETY: the stream and borrowed handles are live for the call.
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

/// Reads numerical values through the package's checked casting path.
fn values(array: Array) -> Vec<f64> {
    let mut output = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the live array and writable output slots satisfy the ABI.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_floats_cast(array, &mut output, &mut length)
        },
        OK
    );
    let values = if length == 0 {
        Vec::new()
    } else {
        // SAFETY: successful nonempty readback exposes `length` initialized values.
        unsafe { std::slice::from_raw_parts(output, length) }.to_vec()
    };
    // SAFETY: copied values are released through their matching destructor.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_float_values(output) };
    values
}

/// Reads numerical values through the package's checked integer casting path.
fn integers(array: Array) -> Vec<i64> {
    let mut output = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the live array and writable output slots satisfy the ABI.
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_ints_cast(array, &mut output, &mut length)
        },
        OK
    );
    let values = if length == 0 {
        Vec::new()
    } else {
        // SAFETY: successful nonempty readback exposes `length` initialized values.
        unsafe { std::slice::from_raw_parts(output, length) }.to_vec()
    };
    // SAFETY: copied values are released through their matching destructor.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_int_values(output) };
    values
}

/// Reads one array's logical shape.
fn shape(array: Array) -> Vec<i64> {
    let mut rank = 0i64;
    let mut dimensions = std::ptr::null_mut();
    // SAFETY: the live array and writable output slots satisfy the ABI.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_rank(array, &mut rank) },
        OK
    );
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_shape(array, &mut dimensions) },
        OK
    );
    if rank == 0 {
        Vec::new()
    } else {
        // SAFETY: successful shape readback exposes `rank` borrowed dimensions.
        unsafe { std::slice::from_raw_parts(dimensions, rank as usize) }.to_vec()
    }
}

/// Reads one array's scalar dtype code.
fn dtype(array: Array) -> i32 {
    let mut output = -1;
    // SAFETY: the live array and writable scalar slot satisfy the ABI.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_dtype(array, &mut output) },
        OK
    );
    output
}

/// Releases one package-owned array.
fn dispose_array(array: Array) {
    // SAFETY: each test calls this exactly once for each owned array.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(array) },
        OK
    );
}

/// Releases one package-owned stream.
fn dispose_stream(stream: Stream) {
    // SAFETY: each test calls this exactly once for each owned stream.
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(stream) },
        OK
    );
}

#[test]
fn one_and_two_parameter_families_broadcast_shapes_domains_and_dtypes() {
    let stream = stream(1201);
    let low = array(&[0.0, 10.0], &[2, 1], 3);
    let high = array(&[10.0, 12.0, 20.0], &[1, 3], 3);
    let uniform = sample(stream, &[low, high], 0, 11);
    let uniform_values = values(uniform);
    assert_eq!(shape(uniform), vec![2, 3]);
    assert_eq!(dtype(uniform), 11);
    for row in 0..2 {
        for column in 0..3 {
            let index = row * 3 + column;
            let lower = [0.0, 10.0][row];
            let upper = [10.0, 12.0, 20.0][column];
            assert!(uniform_values[index] >= lower && uniform_values[index] <= upper);
        }
    }
    assert_eq!(uniform_values[3], 10.0);
    let constant_low = array(&[2145.12, 2145.12], &[2, 1], 3);
    let constant_high = array(&[2145.12, 2145.12], &[1, 2], 3);
    let constant = sample(stream, &[constant_low, constant_high], 0, 3);
    assert_eq!(shape(constant), vec![2, 2]);
    assert_eq!(values(constant), vec![2145.12; 4]);

    let mean = array(&[-3.0, 8.0], &[2, 1], 3);
    let deviation = array(&[0.0, 1.0, 2.0], &[1, 3], 3);
    let normal = sample(stream, &[mean, deviation], 1, 14);
    assert_eq!(shape(normal), vec![2, 3]);
    assert_eq!(dtype(normal), 14);
    let normal_values = values(normal);
    assert_eq!(normal_values[0], -3.0);
    assert_eq!(normal_values[3], 8.0);

    let alpha = array(&[1.0, 2.0], &[2, 1], 3);
    let beta_parameter = array(&[1.0, 3.0, 5.0], &[1, 3], 3);
    let beta = sample(stream, &[alpha, beta_parameter], 2, 3);
    assert_eq!(shape(beta), vec![2, 3]);
    assert!(values(beta).iter().all(|value| (0.0..=1.0).contains(value)));

    let scale = array(&[0.5, 1.0, 2.0], &[1, 3], 3);
    let gamma = sample(stream, &[alpha, scale], 3, 3);
    assert_eq!(shape(gamma), vec![2, 3]);
    assert!(values(gamma).iter().all(|value| *value >= 0.0));

    for array in [
        gamma,
        scale,
        beta,
        beta_parameter,
        alpha,
        normal,
        deviation,
        mean,
        constant,
        constant_high,
        constant_low,
        uniform,
        high,
        low,
    ] {
        dispose_array(array);
    }
    dispose_stream(stream);
}

#[test]
fn power_and_triangular_cover_special_values_and_three_way_broadcasting() {
    let stream = stream(1202);
    let exponent = array(&[2.0, f64::NAN, f64::INFINITY], &[1, 3], 3);
    let power = sample(stream, &[exponent], 4, 3);
    let power_values = values(power);
    assert_eq!(shape(power), vec![1, 3]);
    assert!((0.0..=1.0).contains(&power_values[0]));
    assert!(power_values[1].is_nan());
    assert_eq!(power_values[2], 1.0);

    let left = array(&[0.0, 10.0], &[2, 1], 3);
    let mode = array(&[10.0, 11.0, 15.0], &[1, 3], 3);
    let right = array(&[11.0, 12.0, 20.0], &[1, 3], 3);
    let triangular = sample(stream, &[left, mode, right], 5, 3);
    let triangular_values = values(triangular);
    assert_eq!(shape(triangular), vec![2, 3]);
    for row in 0..2 {
        for column in 0..3 {
            let index = row * 3 + column;
            assert!(triangular_values[index] >= [0.0, 10.0][row]);
            assert!(triangular_values[index] <= [11.0, 12.0, 20.0][column]);
        }
    }

    for array in [triangular, right, mode, left, power, exponent] {
        dispose_array(array);
    }
    dispose_stream(stream);
}

#[test]
fn rejected_broadcast_calls_clear_outputs_and_preserve_stream_state() {
    let tested = stream(1203);
    let control = stream(1203);
    let low = array(&[0.0, 1.0], &[2], 3);
    let high = array(&[2.0, 3.0], &[2], 3);
    let invalid_scale = array(&[1.0, -1.0], &[2], 3);
    let incompatible = array(&[2.0, 3.0, 4.0], &[3], 3);
    let nonfinite_low = array(&[f64::NEG_INFINITY, 0.0], &[2], 3);
    let nonfinite_high = array(&[0.0, f64::INFINITY], &[2], 3);
    let overflow_low = array(&[-f64::MAX], &[], 3);
    let overflow_high = array(&[f64::MAX], &[], 3);
    let reversed_low = array(&[1.0, 2.0], &[2], 3);
    let reversed_high = array(&[1.0, 1.0], &[2], 3);
    let valid_handles = handles(&[low, high]);
    let invalid_handles = handles(&[low, invalid_scale]);
    let incompatible_handles = handles(&[low, incompatible]);
    let mut rejected = 1usize as Array;

    // SAFETY: all pointers are live; each malformed contract must fail before sampling.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_sample_broadcast(
                tested,
                invalid_handles.as_ptr(),
                2,
                1,
                3,
                &mut rejected,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(rejected.is_null());

    for invalid_parameters in [
        [&nonfinite_low, &nonfinite_high],
        [&overflow_low, &overflow_high],
        [&reversed_low, &reversed_high],
    ] {
        rejected = 1usize as Array;
        let invalid_handles = handles(&[*invalid_parameters[0], *invalid_parameters[1]]);
        // SAFETY: each invalid range is materialized in live parameter arrays.
        assert_eq!(
            unsafe {
                terlan_ndarray_v1_random_stream_sample_broadcast(
                    tested,
                    invalid_handles.as_ptr(),
                    2,
                    0,
                    3,
                    &mut rejected,
                )
            },
            INVALID_ARGUMENT
        );
        assert!(rejected.is_null());
    }
    rejected = 1usize as Array;
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_sample_broadcast(
                tested,
                incompatible_handles.as_ptr(),
                2,
                0,
                3,
                &mut rejected,
            )
        },
        native_support::SHAPE_MISMATCH
    );
    assert!(rejected.is_null());

    for (count, distribution, dtype) in [(1, 0, 3), (2, 99, 3), (2, 0, 2)] {
        rejected = 1usize as Array;
        // SAFETY: the malformed scalar metadata is intentionally exercised.
        assert_ne!(
            unsafe {
                terlan_ndarray_v1_random_stream_sample_broadcast(
                    tested,
                    valid_handles.as_ptr(),
                    count,
                    distribution,
                    dtype,
                    &mut rejected,
                )
            },
            OK
        );
        assert!(rejected.is_null());
    }

    let actual = sample(tested, &[low, high], 0, 3);
    let expected = sample(control, &[low, high], 0, 3);
    assert_eq!(values(actual), values(expected));

    for array in [
        expected,
        actual,
        reversed_high,
        reversed_low,
        overflow_high,
        overflow_low,
        nonfinite_high,
        nonfinite_low,
        incompatible,
        invalid_scale,
        high,
        low,
    ] {
        dispose_array(array);
    }
    dispose_stream(control);
    dispose_stream(tested);
}

#[test]
fn non_contiguous_scalar_and_zero_extent_parameters_preserve_logical_shapes() {
    let stream = stream(1204);
    let source = array(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3], 3);
    let mut transposed = std::ptr::null_mut();
    // SAFETY: the source is live and the output slot receives one owned view.
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_transpose(source, 0, 1, &mut transposed) },
        OK
    );
    let high = array(&[100.0], &[], 3);
    let sampled = sample(stream, &[transposed, high], 0, 3);
    assert_eq!(shape(sampled), vec![3, 2]);
    let sampled_values = values(sampled);
    for (value, lower) in sampled_values.iter().zip([0.0, 3.0, 1.0, 4.0, 2.0, 5.0]) {
        assert!(*value >= lower && *value < 100.0);
    }

    let empty = array(&[], &[2, 0, 3], 3);
    let empty_sample = sample(stream, &[empty, high], 0, 11);
    assert_eq!(shape(empty_sample), vec![2, 0, 3]);
    assert_eq!(dtype(empty_sample), 11);
    assert!(values(empty_sample).is_empty());

    let scalar = sample(stream, &[high], 4, 3);
    assert_eq!(shape(scalar), Vec::<i64>::new());
    assert_eq!(values(scalar).len(), 1);

    for array in [
        scalar,
        empty_sample,
        empty,
        sampled,
        high,
        transposed,
        source,
    ] {
        dispose_array(array);
    }
    dispose_stream(stream);
}

#[test]
fn malformed_pointer_contracts_are_rejected_without_dereference() {
    let stream = stream(1205);
    let parameter = array(&[1.0], &[], 3);
    let parameter_handles = handles(&[parameter]);
    let mut output = 1usize as Array;

    // SAFETY: malformed pointers are passed specifically to verify checked rejection.
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_sample_broadcast(
                stream,
                std::ptr::null(),
                1,
                4,
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
            terlan_ndarray_v1_random_stream_sample_broadcast(
                stream,
                parameter_handles.as_ptr(),
                -1,
                4,
                3,
                &mut output,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(output.is_null());
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_sample_broadcast(
                stream,
                parameter_handles.as_ptr(),
                1,
                4,
                3,
                std::ptr::null_mut(),
            )
        },
        INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_sample_broadcast(
                std::ptr::null_mut(),
                parameter_handles.as_ptr(),
                1,
                4,
                3,
                &mut output,
            )
        },
        INVALID_ARGUMENT
    );

    dispose_array(parameter);
    dispose_stream(stream);
}

#[test]
fn remaining_continuous_families_broadcast_through_the_shared_sampler() {
    let stream = stream(1206);
    let rows = array(&[1.0, 2.0], &[2, 1], 3);
    let columns = array(&[1.0, 2.0, 3.0], &[1, 3], 3);
    let locations = array(&[-2.0, 4.0], &[2, 1], 3);
    let scales = array(&[0.5, 1.0, 2.0], &[1, 3], 3);
    let noncentrality = array(&[0.0, 1.0, 2.0], &[1, 3], 3);

    for distribution in [6, 10, 11, 12, 13, 14, 19] {
        let output = sample(stream, &[rows], distribution, 3);
        assert_eq!(shape(output), vec![2, 1]);
        assert_eq!(values(output).len(), 2);
        dispose_array(output);
    }

    for distribution in [7, 8, 9, 15, 16, 20, 21] {
        let parameters = if matches!(distribution, 8 | 9 | 15 | 16 | 21) {
            [locations, scales]
        } else {
            [rows, columns]
        };
        let output = sample(stream, &parameters, distribution, 11);
        assert_eq!(shape(output), vec![2, 3]);
        assert_eq!(dtype(output), 11);
        assert!(values(output).iter().all(|value| value.is_finite()));
        dispose_array(output);
    }

    let noncentral_chisquare = sample(stream, &[rows, noncentrality], 17, 3);
    assert_eq!(shape(noncentral_chisquare), vec![2, 3]);
    assert!(values(noncentral_chisquare)
        .iter()
        .all(|value| *value >= 0.0));
    let noncentral_f = sample(stream, &[rows, columns, noncentrality], 18, 3);
    assert_eq!(shape(noncentral_f), vec![2, 3]);
    assert!(values(noncentral_f).iter().all(|value| *value >= 0.0));

    let zero = array(&[0.0, 0.0, 0.0], &[1, 3], 3);
    for distribution in [8, 15, 16] {
        let constant = sample(stream, &[locations, zero], distribution, 3);
        assert_eq!(values(constant), vec![-2.0, -2.0, -2.0, 4.0, 4.0, 4.0]);
        dispose_array(constant);
    }
    for distribution in [11, 12, 14, 19] {
        let constant = sample(stream, &[zero], distribution, 3);
        assert_eq!(values(constant), vec![0.0; 3]);
        dispose_array(constant);
    }

    for array in [
        zero,
        noncentral_f,
        noncentral_chisquare,
        noncentrality,
        scales,
        locations,
        columns,
        rows,
    ] {
        dispose_array(array);
    }
    dispose_stream(stream);
}

#[test]
fn integer_families_broadcast_shape_dtype_and_count_domains() {
    let stream = stream(1207);
    let rows = array(&[5.0, 10.0], &[2, 1], 3);
    let probabilities = array(&[0.25, 0.5, 1.0], &[1, 3], 3);
    let binomial = sample(stream, &[rows, probabilities], 22, 4);
    assert_eq!(shape(binomial), vec![2, 3]);
    assert_eq!(dtype(binomial), 4);
    assert!(integers(binomial)
        .iter()
        .enumerate()
        .all(|(index, value)| *value >= 0 && *value <= [5, 10][index / 3]));

    let geometric = sample(stream, &[probabilities], 23, 2);
    assert_eq!(shape(geometric), vec![1, 3]);
    assert!(integers(geometric).iter().all(|value| *value >= 1));

    let good = array(&[10.0, 20.0], &[2, 1], 3);
    let bad = array(&[5.0, 10.0, 15.0], &[1, 3], 3);
    let draws = array(&[3.0], &[], 3);
    let hypergeometric = sample(stream, &[good, bad, draws], 24, 2);
    assert_eq!(shape(hypergeometric), vec![2, 3]);
    assert!(integers(hypergeometric)
        .iter()
        .all(|value| (0..=3).contains(value)));

    let rates = array(&[0.0, 4.0], &[2, 1], 3);
    let poisson = sample(stream, &[rates], 25, 2);
    assert_eq!(integers(poisson)[0], 0);
    let exponents = array(&[2.0, 3.0], &[2, 1], 3);
    let zipf = sample(stream, &[exponents], 26, 2);
    assert!(integers(zipf).iter().all(|value| *value >= 1));
    let negative_binomial = sample(stream, &[rows, probabilities], 27, 2);
    assert_eq!(shape(negative_binomial), vec![2, 3]);
    assert_eq!(integers(negative_binomial)[2], 0);
    assert_eq!(integers(negative_binomial)[5], 0);
    let logseries_probability = array(&[0.2, 0.8], &[2, 1], 3);
    let logseries = sample(stream, &[logseries_probability], 28, 2);
    assert!(integers(logseries).iter().all(|value| *value >= 1));

    let scalar_zero = array(&[0.0], &[], 3);
    let scalar_poisson = sample(stream, &[scalar_zero], 25, 2);
    assert_eq!(shape(scalar_poisson), Vec::<i64>::new());
    assert_eq!(integers(scalar_poisson), vec![0]);

    for array in [
        scalar_poisson,
        scalar_zero,
        logseries,
        logseries_probability,
        negative_binomial,
        zipf,
        exponents,
        poisson,
        rates,
        hypergeometric,
        draws,
        bad,
        good,
        geometric,
        binomial,
        probabilities,
        rows,
    ] {
        dispose_array(array);
    }
    dispose_stream(stream);
}

#[test]
fn remaining_broadcast_domains_fail_atomically_without_advancing_the_stream() {
    let tested = stream(1208);
    let control = stream(1208);
    let valid = array(&[1.0, 2.0], &[2], 3);
    let invalid = array(&[1.0, -1.0], &[2], 3);
    let fractional = array(&[3.0, 3.5], &[2], 3);
    let probability = array(&[0.5, 0.5], &[2], 3);
    let impossible_draws = array(&[100.0], &[], 3);
    let invalid_cases = [
        (vec![invalid], 6, 3),
        (vec![fractional, probability], 22, 2),
        (vec![valid, valid, impossible_draws], 24, 2),
        (vec![invalid], 25, 2),
        (vec![valid], 26, 2),
    ];

    for (parameters, distribution, dtype) in invalid_cases {
        let parameter_handles = handles(&parameters);
        let mut output = 1usize as Array;
        // SAFETY: every malformed domain is represented by live parameter arrays.
        assert_eq!(
            unsafe {
                terlan_ndarray_v1_random_stream_sample_broadcast(
                    tested,
                    parameter_handles.as_ptr(),
                    parameter_handles.len() as i64,
                    distribution,
                    dtype,
                    &mut output,
                )
            },
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
    }

    let actual = sample(tested, &[valid], 25, 2);
    let expected = sample(control, &[valid], 25, 2);
    assert_eq!(integers(actual), integers(expected));

    for array in [
        expected,
        actual,
        impossible_draws,
        probability,
        fractional,
        invalid,
        valid,
    ] {
        dispose_array(array);
    }
    dispose_stream(control);
    dispose_stream(tested);
}

#[test]
fn exact_integer_bounds_broadcast_endpoints_shapes_and_dtypes() {
    let stream = stream(1209);
    let low = integer_array(&[-128, 0], &[2, 1], 2);
    let high = integer_array(&[1, 2, 128], &[1, 3], 2);
    let open = sample(stream, &[low, high], 29, 4);
    assert_eq!(shape(open), vec![2, 3]);
    assert_eq!(dtype(open), 4);
    for row in 0..2 {
        for column in 0..3 {
            let value = integers(open)[row * 3 + column];
            assert!(value >= [-128, 0][row]);
            assert!(value < [1, 2, 128][column]);
        }
    }

    let boolean_low = integer_array(&[0], &[], 2);
    let boolean_high = integer_array(&[2], &[], 2);
    let booleans = sample(stream, &[boolean_low, boolean_high], 29, 1);
    assert_eq!(dtype(booleans), 1);
    assert!(matches!(integers(booleans).as_slice(), [0] | [1]));

    let maximum = integer_array(&[i64::MAX], &[], 2);
    let closed = sample(stream, &[maximum, maximum], 30, 2);
    assert_eq!(shape(closed), Vec::<i64>::new());
    assert_eq!(integers(closed), vec![i64::MAX]);

    let empty_low = integer_array(&[], &[3, 0, 4], 2);
    let empty_high = integer_array(&[], &[1, 0, 1], 2);
    let empty = sample(stream, &[empty_low, empty_high], 29, 7);
    assert_eq!(shape(empty), vec![3, 0, 4]);
    assert_eq!(dtype(empty), 7);
    assert!(integers(empty).is_empty());

    for array in [
        empty,
        empty_high,
        empty_low,
        closed,
        maximum,
        booleans,
        boolean_high,
        boolean_low,
        open,
        high,
        low,
    ] {
        dispose_array(array);
    }
    dispose_stream(stream);
}

#[test]
fn exact_integer_bound_failures_are_atomic_and_dtype_checked() {
    let tested = stream(1210);
    let control = stream(1210);
    let low = integer_array(&[0, 1], &[2], 2);
    let high = integer_array(&[2, 3], &[2], 2);
    let reversed = integer_array(&[0, 0], &[2], 2);
    let negative = integer_array(&[-1, 0], &[2], 2);
    let float_bound = array(&[0.0, 1.0], &[2], 3);

    for (parameters, distribution, dtype, status) in [
        (vec![high, reversed], 29, 2, INVALID_ARGUMENT),
        (vec![negative, high], 29, 7, INVALID_ARGUMENT),
        (vec![low, high], 29, 3, UNSUPPORTED_DTYPE),
        (vec![float_bound, high], 29, 2, UNSUPPORTED_DTYPE),
    ] {
        let parameter_handles = handles(&parameters);
        let mut output = 1usize as Array;
        // SAFETY: every malformed contract is represented by live arrays.
        assert_eq!(
            unsafe {
                terlan_ndarray_v1_random_stream_sample_broadcast(
                    tested,
                    parameter_handles.as_ptr(),
                    parameter_handles.len() as i64,
                    distribution,
                    dtype,
                    &mut output,
                )
            },
            status
        );
        assert!(output.is_null());
    }

    let actual = sample(tested, &[low, high], 29, 2);
    let expected = sample(control, &[low, high], 29, 2);
    assert_eq!(integers(actual), integers(expected));

    for array in [expected, actual, float_bound, negative, reversed, high, low] {
        dispose_array(array);
    }
    dispose_stream(control);
    dispose_stream(tested);
}

#[test]
fn scalar_and_explicit_integer_bound_arrays_advance_identical_streams() {
    for distribution in [29, 30] {
        let broadcast_stream = stream(1211 + distribution);
        let explicit_stream = stream(1211 + distribution);
        let low_scalar = integer_array(&[-5], &[], 2);
        let high_vector = integer_array(&[5; 256], &[256], 2);
        let low_vector = integer_array(&[-5; 256], &[256], 2);
        let broadcast = sample(
            broadcast_stream,
            &[low_scalar, high_vector],
            distribution,
            6,
        );
        let explicit = sample(explicit_stream, &[low_vector, high_vector], distribution, 6);
        assert_eq!(shape(broadcast), vec![256]);
        assert_eq!(dtype(broadcast), 6);
        assert_eq!(integers(broadcast), integers(explicit));

        for array in [explicit, broadcast, low_vector, high_vector, low_scalar] {
            dispose_array(array);
        }
        dispose_stream(explicit_stream);
        dispose_stream(broadcast_stream);
    }
}

#[test]
fn single_element_integer_bounds_preserve_every_admitted_output_dtype() {
    let stream = stream(1270);
    let low = integer_array(&[0], &[1], 2);
    let high = integer_array(&[1], &[], 2);
    for dtype_code in [1, 2, 4, 5, 6, 7, 8, 9, 10] {
        let output = sample(stream, &[low, high], 29, dtype_code);
        assert_eq!(shape(output), vec![1]);
        assert_eq!(dtype(output), dtype_code);
        assert_eq!(integers(output), vec![0]);
        dispose_array(output);
    }
    dispose_array(high);
    dispose_array(low);
    dispose_stream(stream);
}
