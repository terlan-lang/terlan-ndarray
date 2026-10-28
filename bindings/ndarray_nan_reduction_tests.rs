//! NumPy 1.21.5 NaN-reduction conformance tests.

use super::*;
use crate::ffi;
use rustfft::num_complex::Complex64;

/// Creates one owned Float64 fixture.
unsafe fn array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, dimensions, 3).unwrap() }
}

/// Creates one owned integer fixture in an explicit dtype.
unsafe fn integer_array(values: &[i64], dimensions: &[i64], dtype: i32) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            ffi::terlan_ndarray_v1_from_ints_as(
                values.as_ptr(),
                values.len() as i64,
                dimensions.as_ptr(),
                dimensions.len() as i64,
                dtype,
                &mut output,
            )
        },
        OK
    );
    output
}

/// Creates one owned complex fixture in an explicit dtype.
unsafe fn complex_array(values: &[Complex64], dimensions: &[i64], dtype: i32) -> Array {
    unsafe { super::super::native_support::owned_complex(values, dimensions, dtype).unwrap() }
}

/// Returns one array's logical values as real scalars.
unsafe fn values(array: Array) -> Vec<f64> {
    unsafe {
        super::super::native_support::numeric(array, false)
            .unwrap()
            .values
            .into_iter()
            .map(|value| value.re)
            .collect()
    }
}

/// Returns one array's logical values as complex scalars.
unsafe fn complex_values(array: Array) -> Vec<Complex64> {
    unsafe {
        super::super::native_support::numeric(array, false)
            .unwrap()
            .values
    }
}

/// Returns one array's logical shape.
unsafe fn dimensions(array: Array) -> Vec<i64> {
    unsafe { super::super::native_support::shape(array).unwrap() }
}

/// Returns one array's exact dtype code.
unsafe fn dtype(array: Array) -> i32 {
    unsafe { super::super::native_support::dtype(array).unwrap() }
}

/// Deletes all supplied arrays.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { super::super::native_support::delete_arrays(arrays) }
}

/// Deletes one output array and clears its handle for safe reuse.
unsafe fn dispose_output(output: &mut Array) {
    unsafe { dispose(&[*output]) };
    *output = std::ptr::null_mut();
}

/// Asserts approximate equality while treating paired NaNs as equal.
fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual.is_nan() && expected.is_nan()) || (actual - expected).abs() <= 1.0e-7,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn nan_reductions_match_axes_keepdims_all_nan_empty_and_mutation_contracts() {
    unsafe {
        let source_values = [
            0.6244,
            f64::NAN,
            0.2692,
            0.0116,
            f64::NAN,
            0.1170,
            0.5351,
            -0.9403,
            f64::NAN,
            0.2100,
            0.4759,
            0.2833,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            0.1042,
            f64::NAN,
            -0.5954,
            0.1610,
            f64::NAN,
            f64::NAN,
            0.1859,
            0.3146,
            f64::NAN,
        ];
        let source = array(&source_values, &[4, 6]);
        let axis = [1_i64];
        let mut output = std::ptr::null_mut();

        assert_eq!(
            ffi::terlan_ndarray_v1_nan_sum(source, axis.as_ptr(), 1, 1, &mut output),
            OK
        );
        assert_eq!(dimensions(output), [4, 1]);
        close(&values(output), &[1.0222, 0.5640, -0.4912, 0.6615]);
        dispose_output(&mut output);

        assert_eq!(
            ffi::terlan_ndarray_v1_nan_product(source, axis.as_ptr(), 1, 0, &mut output),
            OK
        );
        close(
            &values(output),
            &[
                0.6244 * 0.2692 * 0.0116 * 0.1170,
                0.5351 * -0.9403 * 0.2100 * 0.4759 * 0.2833,
                0.1042 * -0.5954,
                0.1610 * 0.1859 * 0.3146,
            ],
        );
        dispose_output(&mut output);

        assert_eq!(
            ffi::terlan_ndarray_v1_nan_mean(source, axis.as_ptr(), 1, 0, &mut output),
            OK
        );
        close(&values(output), &[0.25555, 0.1128, -0.2456, 0.2205]);
        dispose_output(&mut output);

        assert_eq!(
            ffi::terlan_ndarray_v1_nan_minimum(source, axis.as_ptr(), 1, 0, &mut output),
            OK
        );
        close(&values(output), &[0.0116, -0.9403, -0.5954, 0.1610]);
        dispose_output(&mut output);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_maximum(source, axis.as_ptr(), 1, 0, &mut output),
            OK
        );
        close(&values(output), &[0.6244, 0.5351, 0.1042, 0.3146]);
        dispose_output(&mut output);

        assert_eq!(
            ffi::terlan_ndarray_v1_nan_variance(source, axis.as_ptr(), 1, 0, 1, &mut output),
            OK
        );
        close(
            &values(output),
            &[
                0.07164811666666666,
                0.3644291499999999,
                0.24472008,
                0.0067961099999999984,
            ],
        );
        dispose_output(&mut output);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_standard_deviation(
                source,
                axis.as_ptr(),
                1,
                0,
                1,
                &mut output
            ),
            OK
        );
        close(
            &values(output),
            &[
                0.2676716583179225,
                0.6036796749932865,
                0.49469190411810865,
                0.08243852254862406,
            ],
        );
        dispose_output(&mut output);
        for operation in [
            ffi::terlan_ndarray_v1_nan_variance,
            ffi::terlan_ndarray_v1_nan_standard_deviation,
        ] {
            assert_eq!(operation(source, axis.as_ptr(), 1, 0, 6, &mut output), OK);
            close(&values(output), &[f64::NAN; 4]);
            dispose_output(&mut output);
        }

        close(&values(source), &source_values);

        let all_nan = array(&[f64::NAN; 9], &[3, 3]);
        for operation in [
            ffi::terlan_ndarray_v1_nan_minimum,
            ffi::terlan_ndarray_v1_nan_maximum,
            ffi::terlan_ndarray_v1_nan_mean,
        ] {
            assert_eq!(operation(all_nan, axis.as_ptr(), 1, 0, &mut output), OK);
            close(&values(output), &[f64::NAN; 3]);
            dispose_output(&mut output);
        }
        for operation in [
            ffi::terlan_ndarray_v1_nan_variance,
            ffi::terlan_ndarray_v1_nan_standard_deviation,
        ] {
            assert_eq!(operation(all_nan, axis.as_ptr(), 1, 0, 0, &mut output), OK);
            close(&values(output), &[f64::NAN; 3]);
            dispose_output(&mut output);
        }
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_sum(all_nan, std::ptr::null(), 0, 0, &mut output),
            OK
        );
        close(&values(output), &[0.0]);
        dispose_output(&mut output);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_product(all_nan, std::ptr::null(), 0, 0, &mut output),
            OK
        );
        close(&values(output), &[1.0]);
        dispose_output(&mut output);

        let empty = array(&[], &[0, 3]);
        let axis_zero = [0_i64];
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_sum(empty, axis_zero.as_ptr(), 1, 0, &mut output),
            OK
        );
        close(&values(output), &[0.0, 0.0, 0.0]);
        dispose_output(&mut output);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_product(empty, axis_zero.as_ptr(), 1, 0, &mut output),
            OK
        );
        close(&values(output), &[1.0, 1.0, 1.0]);
        dispose_output(&mut output);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_mean(empty, axis_zero.as_ptr(), 1, 0, &mut output),
            OK
        );
        close(&values(output), &[f64::NAN; 3]);
        dispose_output(&mut output);
        for operation in [
            ffi::terlan_ndarray_v1_nan_variance,
            ffi::terlan_ndarray_v1_nan_standard_deviation,
        ] {
            assert_eq!(
                operation(empty, axis_zero.as_ptr(), 1, 0, 0, &mut output),
                OK
            );
            close(&values(output), &[f64::NAN; 3]);
            dispose_output(&mut output);
        }
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_minimum(empty, axis_zero.as_ptr(), 1, 0, &mut output),
            DOMAIN_ERROR
        );
        assert!(output.is_null());
        dispose(&[empty, all_nan, source]);
    }
}

#[test]
fn nan_reductions_preserve_integer_defaults_and_honor_explicit_dtypes() {
    unsafe {
        let data = [127_i64, 39, 93, 87, 46];
        for input_dtype in [2, 4, 5, 6, 7, 8, 9, 10] {
            let source = integer_array(&data, &[5], input_dtype);
            let mut output = std::ptr::null_mut();
            assert_eq!(
                ffi::terlan_ndarray_v1_nan_minimum(source, std::ptr::null(), 0, 0, &mut output),
                OK
            );
            assert_eq!(dtype(output), input_dtype);
            close(&values(output), &[39.0]);
            dispose_output(&mut output);
            assert_eq!(
                ffi::terlan_ndarray_v1_nan_maximum(source, std::ptr::null(), 0, 0, &mut output),
                OK
            );
            assert_eq!(dtype(output), input_dtype);
            close(&values(output), &[127.0]);
            dispose_output(&mut output);
            assert_eq!(
                ffi::terlan_ndarray_v1_nan_sum(source, std::ptr::null(), 0, 0, &mut output),
                OK
            );
            assert_eq!(dtype(output), if input_dtype <= 6 { 2 } else { 10 });
            close(&values(output), &[392.0]);
            dispose_output(&mut output);
            assert_eq!(
                ffi::terlan_ndarray_v1_nan_product(source, std::ptr::null(), 0, 0, &mut output),
                OK
            );
            assert_eq!(dtype(output), if input_dtype <= 6 { 2 } else { 10 });
            close(&values(output), &[1_843_437_258.0]);
            dispose_output(&mut output);
            assert_eq!(
                ffi::terlan_ndarray_v1_nan_mean(source, std::ptr::null(), 0, 0, &mut output),
                OK
            );
            assert_eq!(dtype(output), 3);
            close(&values(output), &[78.4]);
            dispose_output(&mut output);
            assert_eq!(
                ffi::terlan_ndarray_v1_nan_variance(source, std::ptr::null(), 0, 0, 1, &mut output),
                OK
            );
            assert_eq!(dtype(output), 3);
            close(&values(output), &[1312.8]);
            dispose_output(&mut output);
            assert_eq!(
                ffi::terlan_ndarray_v1_nan_standard_deviation(
                    source,
                    std::ptr::null(),
                    0,
                    0,
                    0,
                    &mut output
                ),
                OK
            );
            assert_eq!(dtype(output), 3);
            close(&values(output), &[32.40740656084655]);
            dispose_output(&mut output);
            dispose(&[source]);
        }

        let source = array(&[1.0, f64::NAN, 3.0], &[3]);
        let mut output = std::ptr::null_mut();
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_sum_as(source, std::ptr::null(), 0, 0, 11, &mut output),
            OK
        );
        assert_eq!(dtype(output), 11);
        close(&values(output), &[4.0]);
        dispose_output(&mut output);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_mean_as(source, std::ptr::null(), 0, 0, 11, &mut output),
            OK
        );
        assert_eq!(dtype(output), 11);
        close(&values(output), &[2.0]);
        dispose_output(&mut output);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_mean_as(source, std::ptr::null(), 0, 0, 2, &mut output),
            super::super::native_support::UNSUPPORTED_DTYPE
        );
        assert!(output.is_null());
        dispose(&[source]);

        let complex = complex_array(
            &[
                Complex64::new(1.0, 2.0),
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(3.0, 4.0),
            ],
            &[3],
            13,
        );
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_product_as(complex, std::ptr::null(), 0, 0, 12, &mut output),
            OK
        );
        assert_eq!(dtype(output), 12);
        assert_eq!(complex_values(output), [Complex64::new(-5.0, 10.0)]);
        dispose_output(&mut output);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_variance_as(
                complex,
                std::ptr::null(),
                0,
                0,
                0,
                13,
                &mut output
            ),
            OK
        );
        assert_eq!(dtype(output), 13);
        assert_eq!(complex_values(output), [Complex64::new(2.0, 0.0)]);
        dispose_output(&mut output);
        dispose(&[complex]);
    }
}

#[test]
fn nan_reduction_input_dtypes_follow_numpy_precision_rules() {
    unsafe {
        for input_dtype in [3, 11, 14] {
            let source =
                super::super::native_support::owned_real(&[1.0, f64::NAN, 3.0], &[3], input_dtype)
                    .unwrap();
            let mut output = std::ptr::null_mut();
            for operation in [
                ffi::terlan_ndarray_v1_nan_sum,
                ffi::terlan_ndarray_v1_nan_product,
                ffi::terlan_ndarray_v1_nan_mean,
                ffi::terlan_ndarray_v1_nan_minimum,
                ffi::terlan_ndarray_v1_nan_maximum,
            ] {
                assert_eq!(operation(source, std::ptr::null(), 0, 0, &mut output), OK);
                assert_eq!(dtype(output), input_dtype);
                dispose_output(&mut output);
            }
            for operation in [
                ffi::terlan_ndarray_v1_nan_variance,
                ffi::terlan_ndarray_v1_nan_standard_deviation,
            ] {
                assert_eq!(
                    operation(source, std::ptr::null(), 0, 0, 0, &mut output),
                    OK
                );
                assert_eq!(dtype(output), input_dtype);
                dispose_output(&mut output);
            }
            dispose(&[source]);
        }

        for input_dtype in [12, 13] {
            let source = complex_array(
                &[
                    Complex64::new(1.0, 2.0),
                    Complex64::new(f64::NAN, 0.0),
                    Complex64::new(3.0, 4.0),
                ],
                &[3],
                input_dtype,
            );
            let mut output = std::ptr::null_mut();
            for operation in [
                ffi::terlan_ndarray_v1_nan_sum,
                ffi::terlan_ndarray_v1_nan_product,
                ffi::terlan_ndarray_v1_nan_mean,
                ffi::terlan_ndarray_v1_nan_minimum,
                ffi::terlan_ndarray_v1_nan_maximum,
            ] {
                assert_eq!(operation(source, std::ptr::null(), 0, 0, &mut output), OK);
                assert_eq!(dtype(output), input_dtype);
                dispose_output(&mut output);
            }
            for operation in [
                ffi::terlan_ndarray_v1_nan_variance,
                ffi::terlan_ndarray_v1_nan_standard_deviation,
            ] {
                assert_eq!(
                    operation(source, std::ptr::null(), 0, 0, 0, &mut output),
                    OK
                );
                assert_eq!(dtype(output), if input_dtype == 12 { 11 } else { 3 });
                dispose_output(&mut output);
            }
            dispose(&[source]);
        }

        let source = array(&[1.0, f64::NAN, 3.0], &[3]);
        let mut output = std::ptr::null_mut();
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_sum_as(source, std::ptr::null(), 0, 0, 14, &mut output),
            OK
        );
        assert_eq!(dtype(output), 14);
        dispose_output(&mut output);
        dispose(&[source]);
    }
}

#[test]
fn nan_reduction_destinations_and_axis_arguments_are_exact_and_atomic() {
    unsafe {
        let source = array(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0, 4.0], &[2, 3]);
        let axis = [1_i64];
        let sum_destination = array(&[-1.0, -1.0], &[2]);
        let mean_destination = array(&[-1.0, -1.0], &[2]);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_sum_into(sum_destination, source, axis.as_ptr(), 1, 0),
            OK
        );
        close(&values(sum_destination), &[4.0, 9.0]);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_mean_into(mean_destination, source, axis.as_ptr(), 1, 0),
            OK
        );
        close(&values(mean_destination), &[2.0, 4.5]);

        let wrong_shape = array(&[17.0], &[1]);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_sum_into(wrong_shape, source, axis.as_ptr(), 1, 0),
            super::super::native_support::SHAPE_MISMATCH
        );
        close(&values(wrong_shape), &[17.0]);
        let integer_destination = integer_array(&[17, 18], &[2], 2);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_mean_into(integer_destination, source, axis.as_ptr(), 1, 0),
            super::super::native_support::UNSUPPORTED_DTYPE
        );
        close(&values(integer_destination), &[17.0, 18.0]);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_variance_into(
                integer_destination,
                source,
                axis.as_ptr(),
                1,
                0,
                0
            ),
            super::super::native_support::UNSUPPORTED_DTYPE
        );
        close(&values(integer_destination), &[17.0, 18.0]);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_standard_deviation_into(
                integer_destination,
                source,
                axis.as_ptr(),
                1,
                0,
                0
            ),
            super::super::native_support::UNSUPPORTED_DTYPE
        );
        close(&values(integer_destination), &[17.0, 18.0]);

        let mut indexes = std::ptr::null_mut();
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_argmin(source, axis.as_ptr(), 1, 0, &mut indexes),
            OK
        );
        close(&values(indexes), &[0.0, 2.0]);
        dispose(&[indexes]);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_argmax(source, axis.as_ptr(), 1, 1, &mut indexes),
            OK
        );
        assert_eq!(dimensions(indexes), [2, 1]);
        close(&values(indexes), &[2.0, 1.0]);
        dispose(&[indexes]);

        let all_nan = array(&[f64::NAN; 4], &[2, 2]);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_argmin(all_nan, axis.as_ptr(), 1, 0, &mut indexes),
            DOMAIN_ERROR
        );
        assert!(indexes.is_null());
        let empty = array(&[], &[0, 3]);
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_argmax(empty, axis.as_ptr(), 1, 0, &mut indexes),
            OK
        );
        assert_eq!(dimensions(indexes), [0]);
        dispose(&[indexes]);
        let axis_zero = [0_i64];
        assert_eq!(
            ffi::terlan_ndarray_v1_nan_argmax(empty, axis_zero.as_ptr(), 1, 0, &mut indexes),
            DOMAIN_ERROR
        );
        assert!(indexes.is_null());
        dispose(&[
            empty,
            all_nan,
            integer_destination,
            wrong_shape,
            mean_destination,
            sum_destination,
            source,
        ]);
    }
}

#[test]
fn nan_cumulative_operations_cover_axes_flattening_dtypes_outputs_and_empty_shapes() {
    unsafe {
        let source = array(&[1.0, f64::NAN, 2.0, f64::NAN, 3.0, 4.0], &[2, 3]);
        let mut output = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_nancumsum(source, 1, &mut output), OK);
        close(&values(output), &[1.0, 1.0, 3.0, 0.0, 3.0, 7.0]);
        dispose_output(&mut output);
        assert_eq!(terlan_ndarray_v1_nancumprod(source, -1, &mut output), OK);
        close(&values(output), &[1.0, 1.0, 2.0, 1.0, 3.0, 12.0]);
        dispose_output(&mut output);
        assert_eq!(terlan_ndarray_v1_nancumsum_flat(source, &mut output), OK);
        assert_eq!(dimensions(output), [6]);
        close(&values(output), &[1.0, 1.0, 3.0, 3.0, 6.0, 10.0]);
        dispose_output(&mut output);
        assert_eq!(terlan_ndarray_v1_nancumprod_flat(source, &mut output), OK);
        close(&values(output), &[1.0, 1.0, 2.0, 2.0, 6.0, 24.0]);
        dispose_output(&mut output);
        assert_eq!(
            terlan_ndarray_v1_nancumsum_as(source, 1, 11, &mut output),
            OK
        );
        assert_eq!(dtype(output), 11);
        dispose_output(&mut output);

        let destination = array(&[0.0; 6], &[2, 3]);
        assert_eq!(terlan_ndarray_v1_nancumsum_into(destination, source, 1), OK);
        close(&values(destination), &[1.0, 1.0, 3.0, 0.0, 3.0, 7.0]);
        let flat_destination = array(&[0.0; 6], &[6]);
        assert_eq!(
            terlan_ndarray_v1_nancumprod_flat_into(flat_destination, source),
            OK
        );
        close(&values(flat_destination), &[1.0, 1.0, 2.0, 2.0, 6.0, 24.0]);
        let wrong_destination = array(&[17.0], &[1]);
        assert_eq!(
            terlan_ndarray_v1_nancumsum_into(wrong_destination, source, 1),
            super::super::native_support::SHAPE_MISMATCH
        );
        close(&values(wrong_destination), &[17.0]);
        close(&values(source), &[1.0, f64::NAN, 2.0, f64::NAN, 3.0, 4.0]);

        let all_nan = array(&[f64::NAN; 3], &[3]);
        assert_eq!(terlan_ndarray_v1_nancumsum_flat(all_nan, &mut output), OK);
        close(&values(output), &[0.0, 0.0, 0.0]);
        dispose_output(&mut output);
        assert_eq!(terlan_ndarray_v1_nancumprod_flat(all_nan, &mut output), OK);
        close(&values(output), &[1.0, 1.0, 1.0]);
        dispose_output(&mut output);

        for input_dtype in [2, 4, 5, 6, 7, 8, 9, 10] {
            let integer = integer_array(&[127, 39, 93, 87, 46], &[5], input_dtype);
            assert_eq!(terlan_ndarray_v1_nancumsum(integer, 0, &mut output), OK);
            assert_eq!(dtype(output), if input_dtype <= 6 { 2 } else { 10 });
            close(&values(output), &[127.0, 166.0, 259.0, 346.0, 392.0]);
            dispose_output(&mut output);
            assert_eq!(terlan_ndarray_v1_nancumprod(integer, 0, &mut output), OK);
            assert_eq!(dtype(output), if input_dtype <= 6 { 2 } else { 10 });
            close(
                &values(output),
                &[127.0, 4_953.0, 460_629.0, 40_074_723.0, 1_843_437_258.0],
            );
            dispose_output(&mut output);
            dispose(&[integer]);
        }

        let empty = array(&[], &[0, 3]);
        assert_eq!(terlan_ndarray_v1_nancumsum(empty, 0, &mut output), OK);
        assert_eq!(dimensions(output), [0, 3]);
        assert!(values(output).is_empty());
        dispose_output(&mut output);
        assert_eq!(terlan_ndarray_v1_nancumprod_flat(empty, &mut output), OK);
        assert_eq!(dimensions(output), [0]);
        assert!(values(output).is_empty());
        dispose_output(&mut output);
        dispose(&[
            empty,
            all_nan,
            wrong_destination,
            flat_destination,
            destination,
            source,
        ]);
    }
}
