//! NumPy-derived regression tests for portable random-stream behavior.

use super::*;

type Array = *mut super::super::ffi::TerlanNdArray;
type Stream = *mut super::super::ffi::TerlanNdRandomStream;

/// Creates one deterministic owned random stream.
fn stream(seed: i64) -> Stream {
    let mut output = std::ptr::null_mut();
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_new(seed, &mut output) },
        OK
    );
    output
}

/// Creates one owned numerical array.
fn real_array(values: &[f64], shape: &[i64], dtype: i32) -> Array {
    let mut output = std::ptr::null_mut();
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

/// Creates one owned integer array.
fn integer_array(values: &[i64], shape: &[i64]) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_from_ints(
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

/// Reads one array through the checked floating conversion path.
fn floats(array: Array) -> Vec<f64> {
    let mut output = std::ptr::null_mut();
    let mut length = 0_usize;
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_floats_cast(array, &mut output, &mut length)
        },
        OK
    );
    let values = if length == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(output, length) }.to_vec()
    };
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_float_values(output) };
    values
}

/// Reads one array through the checked integer conversion path.
fn integers(array: Array) -> Vec<i64> {
    let mut output = std::ptr::null_mut();
    let mut length = 0_usize;
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_to_ints_cast(array, &mut output, &mut length)
        },
        OK
    );
    let values = if length == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(output, length) }.to_vec()
    };
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_int_values(output) };
    values
}

/// Releases one owned array.
fn dispose_array(array: Array) {
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_delete(array) },
        OK
    );
}

/// Releases one owned stream.
fn dispose_stream(stream: Stream) {
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_delete(stream) },
        OK
    );
}

/// Exercises numerical stability and domain regressions independent of RNG choice.
#[test]
fn small_parameters_zero_shapes_and_distribution_ranges_remain_stable() {
    let stream = stream(1234567890);
    let shape = [100_i64];
    let mut output = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_beta(
                stream,
                shape.as_ptr(),
                1,
                0.0001,
                0.0001,
                3,
                &mut output,
            )
        },
        OK
    );
    assert!(floats(output).iter().all(|value| value.is_finite()));
    dispose_array(output);

    let singleton = [1_i64];
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_standard_gamma(
                stream,
                singleton.as_ptr(),
                1,
                0.0,
                11,
                &mut output,
            )
        },
        OK
    );
    assert_eq!(floats(output), [0.0]);
    dispose_array(output);

    let vonmises_shape = [50_i64];
    for mean in [-7.0, -3.5, 0.0, 3.5, 7.0] {
        assert_eq!(
            unsafe {
                terlan_ndarray_v1_random_stream_vonmises(
                    stream,
                    vonmises_shape.as_ptr(),
                    1,
                    mean,
                    1.0,
                    3,
                    &mut output,
                )
            },
            OK
        );
        assert!(floats(output)
            .iter()
            .all(|value| *value > -std::f64::consts::PI && *value <= std::f64::consts::PI));
        dispose_array(output);
    }

    let ten = [10_i64];
    for (good, bad, samples, upper, lower) in [(3, 18, 11, 4, -1), (18, 3, 11, i64::MAX, 0)] {
        assert_eq!(
            unsafe {
                terlan_ndarray_v1_random_stream_hypergeometric(
                    stream,
                    ten.as_ptr(),
                    1,
                    good,
                    bad,
                    samples,
                    2,
                    &mut output,
                )
            },
            OK
        );
        assert!(integers(output)
            .iter()
            .all(|value| *value < upper && *value > lower));
        dispose_array(output);
    }
    let large = (1_i64 << 20) - 2;
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_hypergeometric(
                stream,
                std::ptr::null(),
                0,
                large,
                large,
                large,
                2,
                &mut output,
            )
        },
        OK
    );
    assert!(integers(output)[0] > 0);
    dispose_array(output);
    dispose_stream(stream);
}

/// Keeps log-series frequencies and typed choice probability tolerances useful.
#[test]
fn logseries_converges_and_choice_accepts_normalized_low_precision_weights() {
    let stream = stream(0);
    let shape = [1000_i64];
    let mut output = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_logseries(
                stream,
                shape.as_ptr(),
                1,
                0.8,
                2,
                &mut output,
            )
        },
        OK
    );
    let samples = integers(output);
    let ones = samples.iter().filter(|value| **value == 1).count() as f64 / 1000.0;
    let twos = samples.iter().filter(|value| **value == 2).count() as f64 / 1000.0;
    assert!(ones > 0.45);
    assert!(twos < 0.23);
    dispose_array(output);

    let source = integer_array(&[1, 2, 3], &[3]);
    for dtype in [14, 11, 3] {
        let probabilities = real_array(&[0.4, 0.4, 0.2], &[3], dtype);
        assert_eq!(
            unsafe {
                terlan_ndarray_v1_random_stream_choice(
                    stream,
                    source,
                    std::ptr::null(),
                    0,
                    probabilities,
                    1,
                    &mut output,
                )
            },
            OK
        );
        assert!([1, 2, 3].contains(&integers(output)[0]));
        dispose_array(output);
        dispose_array(probabilities);
    }
    let invalid = real_array(&[0.36, 0.36, 0.18], &[3], 3);
    output = 1_usize as Array;
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_choice(
                stream,
                source,
                std::ptr::null(),
                0,
                invalid,
                1,
                &mut output,
            )
        },
        INVALID_ARGUMENT
    );
    assert!(output.is_null());
    dispose_array(invalid);
    dispose_array(source);
    dispose_stream(stream);
}

/// Returns a deterministic sequence generated entirely inside one thread.
fn threaded_sequence(seed: i64) -> (Vec<f64>, Vec<f64>, Vec<i64>) {
    let stream = stream(seed);
    let shape = [64_i64];
    let mut normal = std::ptr::null_mut();
    let mut exponential = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_standard_normal(
                stream,
                shape.as_ptr(),
                1,
                3,
                &mut normal,
            )
        },
        OK
    );
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_standard_exponential(
                stream,
                shape.as_ptr(),
                1,
                3,
                &mut exponential,
            )
        },
        OK
    );
    let probabilities = real_array(&[0.2, 0.3, 0.5], &[3], 3);
    let batches = [8_i64];
    let mut multinomial = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_multinomial(
                stream,
                10,
                probabilities,
                batches.as_ptr(),
                1,
                2,
                &mut multinomial,
            )
        },
        OK
    );
    let result = (floats(normal), floats(exponential), integers(multinomial));
    dispose_array(multinomial);
    dispose_array(probabilities);
    dispose_array(exponential);
    dispose_array(normal);
    dispose_stream(stream);
    result
}

/// Matches independent stream results between concurrent and serial execution.
#[test]
fn independent_random_streams_are_thread_safe_and_reproducible() {
    let seeds = [0_i64, 1, 2, 3];
    let expected = seeds.map(threaded_sequence);
    let workers = seeds.map(|seed| std::thread::spawn(move || threaded_sequence(seed)));
    let actual = workers.map(|worker| worker.join().expect("random worker must not panic"));
    assert_eq!(actual, expected);
}

/// Matches allocated results while rejecting invalid destinations atomically.
#[test]
fn destination_fills_match_direct_sampling_and_preserve_streams_on_failure() {
    for dtype in [3, 11] {
        let shape = [3_i64, 4];
        for (seed, fill, distribution) in [
            (
                301_i64,
                terlan_ndarray_v1_random_stream_uniform_into
                    as unsafe extern "C" fn(Stream, Array) -> i32,
                0_i32,
            ),
            (
                302_i64,
                terlan_ndarray_v1_random_stream_standard_normal_into,
                1_i32,
            ),
            (
                303_i64,
                terlan_ndarray_v1_random_stream_standard_exponential_into,
                2_i32,
            ),
        ] {
            let tested = stream(seed);
            let control = stream(seed);
            let destination = real_array(&[0.0; 12], &shape, dtype);
            let mut expected = std::ptr::null_mut();
            assert_eq!(unsafe { fill(tested, destination) }, OK);
            let status = unsafe {
                match distribution {
                    0 => terlan_ndarray_v1_random_stream_uniform(
                        control,
                        shape.as_ptr(),
                        2,
                        0.0,
                        1.0,
                        dtype,
                        &mut expected,
                    ),
                    1 => terlan_ndarray_v1_random_stream_standard_normal(
                        control,
                        shape.as_ptr(),
                        2,
                        dtype,
                        &mut expected,
                    ),
                    _ => terlan_ndarray_v1_random_stream_standard_exponential(
                        control,
                        shape.as_ptr(),
                        2,
                        dtype,
                        &mut expected,
                    ),
                }
            };
            assert_eq!(status, OK);
            assert_eq!(floats(destination), floats(expected));
            dispose_array(expected);
            dispose_array(destination);
            dispose_stream(control);
            dispose_stream(tested);
        }
    }

    let tested = stream(304);
    let control = stream(304);
    let destination = real_array(&[0.0; 6], &[2, 3], 11);
    let gamma_shapes = real_array(&[1.0, 2.0, 3.0], &[3], 11);
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_standard_gamma_into(tested, destination, gamma_shapes)
        },
        OK
    );
    let mut broadcast = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            super::super::ffi::terlan_ndarray_v1_broadcast_to(
                gamma_shapes,
                [2_i64, 3].as_ptr(),
                2,
                &mut broadcast,
            )
        },
        OK
    );
    let parameters = [broadcast as usize as u64];
    let mut expected = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_random_stream_sample_broadcast(
                control,
                parameters.as_ptr(),
                1,
                12,
                11,
                &mut expected,
            )
        },
        OK
    );
    assert_eq!(floats(destination), floats(expected));
    dispose_array(expected);
    dispose_array(broadcast);
    dispose_array(gamma_shapes);
    dispose_array(destination);
    dispose_stream(control);
    dispose_stream(tested);

    let tested = stream(305);
    let control = stream(305);
    let integer_destination = integer_array(&[0, 0, 0, 0], &[2, 2]);
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_uniform_into(tested, integer_destination) },
        109
    );
    let mut actual = std::ptr::null_mut();
    let mut expected = std::ptr::null_mut();
    for (stream, output) in [(tested, &mut actual), (control, &mut expected)] {
        assert_eq!(
            unsafe {
                terlan_ndarray_v1_random_stream_uniform(
                    stream,
                    [4_i64].as_ptr(),
                    1,
                    0.0,
                    1.0,
                    3,
                    output,
                )
            },
            OK
        );
    }
    assert_eq!(floats(actual), floats(expected));

    let storage = real_array(&[0.0; 6], &[2, 3], 3);
    let mut transposed = std::ptr::null_mut();
    assert_eq!(
        unsafe { super::super::ffi::terlan_ndarray_v1_transpose(storage, 0, 1, &mut transposed) },
        OK
    );
    assert_eq!(
        unsafe { terlan_ndarray_v1_random_stream_standard_normal_into(tested, transposed) },
        123
    );
    assert_eq!(floats(storage), [0.0; 6]);
    dispose_array(transposed);
    dispose_array(storage);
    dispose_array(expected);
    dispose_array(actual);
    dispose_array(integer_destination);
    dispose_stream(control);
    dispose_stream(tested);
}

/// Detects modulo bias across every small integer storage width.
#[test]
fn small_integer_dtypes_remain_statistically_uniform() {
    let stream = stream(1234567890);
    for (sample_count, upper, dtype, maximum_chi_squared) in [
        (500_000_i64, 5_i64, 4_i32, 125.0_f64),
        (500_000, 7, 7, 150.0),
        (1_000_000, 2_500, 5, 3_300.0),
        (1_000_000, 5_000, 8, 6_500.0),
    ] {
        let shape = [sample_count];
        let mut output = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                terlan_ndarray_v1_random_stream_integers(
                    stream,
                    shape.as_ptr(),
                    1,
                    0,
                    upper,
                    dtype,
                    &mut output,
                )
            },
            OK
        );
        let mut counts = vec![0_u64; upper as usize];
        for value in integers(output) {
            assert!((0..upper).contains(&value));
            counts[value as usize] += 1;
        }
        let expected = sample_count as f64 / upper as f64;
        let chi_squared = counts
            .iter()
            .map(|count| {
                let difference = *count as f64 - expected;
                difference * difference / expected
            })
            .sum::<f64>();
        assert!(
            chi_squared < maximum_chi_squared,
            "dtype {dtype} over [0, {upper}) produced chi-squared {chi_squared}"
        );
        dispose_array(output);
    }
    dispose_stream(stream);
}
