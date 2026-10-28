//! NumPy 1.21.5 tensor polynomial completion tests.

use super::super::polynomial_extension::{
    terlan_ndarray_v1_polyadd, terlan_ndarray_v1_polydiv_all, terlan_ndarray_v1_polymul,
    terlan_ndarray_v1_polyroots,
};
use super::*;

const OK: i32 = 0;

/// Creates one owned Float64 fixture.
unsafe fn array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, dimensions, 3).unwrap() }
}

/// Creates one owned Complex128 fixture.
unsafe fn complex_array(values: &[Complex64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_complex(values, dimensions, 13).unwrap() }
}

/// Creates one owned Int64 fixture.
unsafe fn int_array(values: &[i64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_int(values, dimensions).unwrap() }
}

/// Reads one result's logical numeric values.
unsafe fn result_values(array: Array) -> Vec<Complex64> {
    unsafe {
        super::super::native_support::numeric(array, false)
            .unwrap()
            .values
    }
}

/// Returns one result's shape.
unsafe fn dimensions(array: Array) -> Vec<i64> {
    unsafe { shape(array).unwrap() }
}

/// Deletes all supplied arrays.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { super::super::native_support::delete_arrays(arrays) }
}

/// Takes ownership of arrays returned through one list-valued native operation.
unsafe fn returned_arrays(raw: *mut u64, length: usize) -> Vec<Array> {
    let arrays = unsafe { std::slice::from_raw_parts(raw, length) }
        .iter()
        .map(|identity| *identity as usize as Array)
        .collect::<Vec<_>>();
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(raw) };
    arrays
}

/// Asserts approximate complex equality.
fn close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).norm() <= 1.0e-7,
            "{actual:?} != {expected:?}"
        );
    }
}

/// Converts real values to the complex test representation.
fn real(values: &[f64]) -> Vec<Complex64> {
    values
        .iter()
        .map(|value| Complex64::new(*value, 0.0))
        .collect()
}

#[test]
fn numpy_polyder_supports_scaling_and_arbitrary_axes() {
    unsafe {
        let coefficients = array(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        );
        let mut axis_zero = std::ptr::null_mut();
        let mut axis_one = std::ptr::null_mut();
        let mut exhausted = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyder_axis(coefficients, 1, 1.0, 0, &mut axis_zero),
            OK
        );
        assert_eq!(dimensions(axis_zero), [2, 4]);
        close(
            &result_values(axis_zero),
            &real(&[5.0, 6.0, 7.0, 8.0, 18.0, 20.0, 22.0, 24.0]),
        );
        assert_eq!(
            terlan_ndarray_v1_polyder_axis(coefficients, 1, 0.5, -1, &mut axis_one),
            OK
        );
        assert_eq!(dimensions(axis_one), [3, 3]);
        close(
            &result_values(axis_one),
            &real(&[1.0, 3.0, 6.0, 3.0, 7.0, 12.0, 5.0, 11.0, 18.0]),
        );
        assert_eq!(
            terlan_ndarray_v1_polyder_axis(coefficients, 8, 1.0, 1, &mut exhausted),
            OK
        );
        assert_eq!(dimensions(exhausted), [3, 1]);
        close(&result_values(exhausted), &real(&[0.0, 0.0, 0.0]));
        dispose(&[exhausted, axis_one, axis_zero, coefficients]);
    }
}

#[test]
fn numpy_polyint_supports_constants_bounds_scaling_and_axes() {
    unsafe {
        let coefficients = array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let constants = array(&[5.0], &[1]);
        let mut integrated = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyint_axis(
                coefficients,
                1,
                constants,
                -1.0,
                2.0,
                0,
                &mut integrated,
            ),
            OK
        );
        assert_eq!(dimensions(integrated), [3, 2]);
        close(
            &result_values(integrated),
            &real(&[4.0, 5.0, 2.0, 4.0, 3.0, 4.0]),
        );

        let zero = array(&[0.0], &[1]);
        let staged_constants = array(&[0.0, 1.0], &[2]);
        let mut repeated = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyint_axis(zero, 3, staged_constants, 0.0, 1.0, -1, &mut repeated,),
            OK
        );
        assert_eq!(dimensions(repeated), [2]);
        close(&result_values(repeated), &real(&[0.0, 1.0]));
        dispose(&[
            repeated,
            staged_constants,
            zero,
            integrated,
            constants,
            coefficients,
        ]);
    }
}

#[test]
fn numpy_polyvalfromroots_supports_tensor_and_broadcast_modes() {
    unsafe {
        let coordinates = array(&[0.0, 1.0], &[2]);
        let roots = array(&[0.0, 1.0, 2.0, 3.0], &[2, 2]);
        let mut tensor = std::ptr::null_mut();
        let mut pointwise = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyvalfromroots_tensor(coordinates, roots, 1, &mut tensor),
            OK
        );
        assert_eq!(dimensions(tensor), [2, 2]);
        close(&result_values(tensor), &real(&[0.0, -1.0, 3.0, 0.0]));
        assert_eq!(
            terlan_ndarray_v1_polyvalfromroots_tensor(coordinates, roots, 0, &mut pointwise),
            OK
        );
        assert_eq!(dimensions(pointwise), [2]);
        close(&result_values(pointwise), &real(&[0.0, 0.0]));

        let empty = array(&[], &[0]);
        let root_grid = array(&[1.0, 1.0, 1.0, 1.0, 1.0], &[1, 5]);
        let mut empty_result = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyvalfromroots_tensor(empty, root_grid, 1, &mut empty_result),
            OK
        );
        assert_eq!(dimensions(empty_result), [5, 0]);
        assert!(result_values(empty_result).is_empty());

        let roots_vector = array(&[1.0], &[1]);
        let mut invalid = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyvalfromroots_tensor(coordinates, roots_vector, 0, &mut invalid,),
            SHAPE_MISMATCH
        );
        assert!(invalid.is_null());
        dispose(&[
            roots_vector,
            empty_result,
            root_grid,
            empty,
            pointwise,
            tensor,
            roots,
            coordinates,
        ]);
    }
}

#[test]
fn numpy_polyfit_supports_selected_degrees_weights_and_matrix_values() {
    unsafe {
        let x_values = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
        let y_values = x_values
            .iter()
            .map(|x| 1.0 + x * x + x * x * x * x)
            .collect::<Vec<_>>();
        let matrix_values = y_values
            .iter()
            .flat_map(|value| [*value, *value])
            .collect::<Vec<_>>();
        let coordinates = array(&x_values, &[9]);
        let values = array(&matrix_values, &[9, 2]);
        let degrees = int_array(&[0, 2, 4], &[3]);
        let mut fitted = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyfit_degrees(coordinates, values, degrees, &mut fitted),
            OK
        );
        assert_eq!(dimensions(fitted), [5, 2]);
        close(
            &result_values(fitted),
            &real(&[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
        );

        let cubic = x_values
            .iter()
            .map(|x| x * (x - 1.0) * (x - 2.0))
            .collect::<Vec<_>>();
        let weighted_values = array(&cubic, &[9]);
        let all_degrees = int_array(&[0, 1, 2, 3], &[4]);
        let weights = array(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[9]);
        let mut weighted = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyfit_weighted_degrees(
                coordinates,
                weighted_values,
                all_degrees,
                weights,
                &mut weighted,
            ),
            OK
        );
        close(&result_values(weighted), &real(&[0.0, 2.0, -3.0, 1.0]));
        dispose(&[
            weighted,
            weights,
            all_degrees,
            weighted_values,
            fitted,
            degrees,
            values,
            coordinates,
        ]);
    }
}

#[test]
fn numpy_polyfit_supports_complex_coordinates() {
    unsafe {
        let coordinates = complex_array(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, -1.0),
            ],
            &[4],
        );
        let values = complex_array(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, -1.0),
            ],
            &[4],
        );
        let degrees = int_array(&[0, 1], &[2]);
        let mut fitted = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyfit_degrees(coordinates, values, degrees, &mut fitted),
            OK
        );
        close(
            &result_values(fitted),
            &[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        );
        dispose(&[fitted, degrees, values, coordinates]);
    }
}

#[test]
fn numpy_polyfit_covariance_matches_scaled_weighted_and_matrix_contracts() {
    unsafe {
        let coordinates = array(
            &[0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0],
            &[7],
        );
        let observations = array(&[2.0, 1.0, 14.0 / 3.0, 5.0, 10.0, 35.0 / 3.0, 18.0], &[7]);
        let weights = array(
            &[
                64.0 / 7.0,
                7.0,
                36.0 / 7.0,
                25.0 / 7.0,
                16.0 / 7.0,
                9.0 / 7.0,
                4.0 / 7.0,
            ],
            &[7],
        );

        let mut outputs = std::ptr::null_mut();
        let mut output_length = 0;
        assert_eq!(
            terlan_ndarray_v1_polyfit_covariance(
                coordinates,
                observations,
                2,
                1,
                &mut outputs,
                &mut output_length,
            ),
            OK
        );
        assert_eq!(output_length, 2);
        let scaled = returned_arrays(outputs, output_length);
        assert_eq!(dimensions(scaled[0]), [3]);
        assert_eq!(dimensions(scaled[1]), [3, 3]);
        close(
            &result_values(scaled[0]),
            &real(&[1.61904762, 0.28571429, 3.85714286]),
        );
        close(
            &result_values(scaled[1]),
            &real(&[
                1.16099773,
                -2.12244898,
                0.81632653,
                -2.12244898,
                6.36734694,
                -2.93877551,
                0.81632653,
                -2.93877551,
                1.46938776,
            ]),
        );

        outputs = std::ptr::null_mut();
        output_length = 0;
        assert_eq!(
            terlan_ndarray_v1_polyfit_weighted_covariance(
                coordinates,
                observations,
                2,
                weights,
                1,
                &mut outputs,
                &mut output_length,
            ),
            OK
        );
        let weighted_scaled = returned_arrays(outputs, output_length);
        close(
            &result_values(weighted_scaled[0]),
            &real(&[1.77675979, -1.01769136, 4.89274172]),
        );
        close(
            &result_values(weighted_scaled[1]),
            &real(&[
                0.33367938,
                -0.90891376,
                0.48778036,
                -0.90891376,
                6.80672276,
                -5.00516045,
                0.48778036,
                -5.00516045,
                4.39644554,
            ]),
        );

        outputs = std::ptr::null_mut();
        output_length = 0;
        assert_eq!(
            terlan_ndarray_v1_polyfit_weighted_covariance(
                coordinates,
                observations,
                2,
                weights,
                0,
                &mut outputs,
                &mut output_length,
            ),
            OK
        );
        let weighted_unscaled = returned_arrays(outputs, output_length);
        close(
            &result_values(weighted_unscaled[1]),
            &real(&[
                0.01117754,
                -0.03044666,
                0.01633960,
                -0.03044666,
                0.22801064,
                -0.16766216,
                0.01633960,
                -0.16766216,
                0.14727151,
            ]),
        );

        let matrix_observations = array(
            &[
                2.0,
                2.0,
                1.0,
                1.0,
                14.0 / 3.0,
                14.0 / 3.0,
                5.0,
                5.0,
                10.0,
                10.0,
                35.0 / 3.0,
                35.0 / 3.0,
                18.0,
                18.0,
            ],
            &[7, 2],
        );
        outputs = std::ptr::null_mut();
        output_length = 0;
        assert_eq!(
            terlan_ndarray_v1_polyfit_covariance(
                coordinates,
                matrix_observations,
                2,
                1,
                &mut outputs,
                &mut output_length,
            ),
            OK
        );
        let matrix = returned_arrays(outputs, output_length);
        assert_eq!(dimensions(matrix[0]), [3, 2]);
        assert_eq!(dimensions(matrix[1]), [3, 3, 2]);
        for pair in result_values(matrix[0]).chunks_exact(2) {
            close(pair, &[pair[0], pair[0]]);
        }
        for pair in result_values(matrix[1]).chunks_exact(2) {
            close(pair, &[pair[0], pair[0]]);
        }

        dispose(&matrix);
        dispose(&[matrix_observations]);
        dispose(&weighted_unscaled);
        dispose(&weighted_scaled);
        dispose(&scaled);
        dispose(&[weights, observations, coordinates]);
    }
}

#[test]
fn numpy_polynomial_calculus_promotes_integers_and_preserves_complex_values() {
    unsafe {
        let integer = int_array(&[1, 1, 1], &[3]);
        let complex = complex_array(
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
            ],
            &[3],
        );
        let constants = array(&[], &[0]);
        let mut integer_integral = std::ptr::null_mut();
        let mut complex_integral = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyint_axis(
                integer,
                1,
                constants,
                0.0,
                1.0,
                0,
                &mut integer_integral,
            ),
            OK
        );
        assert_eq!(
            terlan_ndarray_v1_polyint_axis(
                complex,
                1,
                constants,
                0.0,
                1.0,
                0,
                &mut complex_integral,
            ),
            OK
        );
        assert_eq!(
            super::super::native_support::dtype(integer_integral).unwrap(),
            3
        );
        assert_eq!(
            super::super::native_support::dtype(complex_integral).unwrap(),
            13
        );
        close(
            &result_values(integer_integral),
            &real(&[0.0, 1.0, 0.5, 1.0 / 3.0]),
        );
        close(
            &result_values(complex_integral),
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(1.0, -0.5),
                Complex64::new(1.0, 1.0 / 6.0),
            ],
        );
        dispose(&[
            complex_integral,
            integer_integral,
            constants,
            complex,
            integer,
        ]);
    }
}

#[test]
fn numpy_polyfit_remains_stable_for_large_offset_coordinates() {
    unsafe {
        let x = [
            90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0,
            104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0,
            116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 129.0,
            130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0, 141.0,
            146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0,
            158.0, 159.0, 160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0,
            170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0,
        ];
        let y = [
            9.0, 3.0, 7.0, 4.0, 4.0, 8.0, 6.0, 11.0, 9.0, 8.0, 11.0, 5.0, 6.0, 5.0, 9.0, 8.0, 6.0,
            10.0, 6.0, 10.0, 7.0, 6.0, 6.0, 6.0, 13.0, 4.0, 9.0, 11.0, 4.0, 5.0, 8.0, 5.0, 7.0,
            7.0, 6.0, 12.0, 7.0, 7.0, 9.0, 4.0, 12.0, 6.0, 6.0, 4.0, 3.0, 9.0, 8.0, 8.0, 6.0, 7.0,
            9.0, 10.0, 6.0, 8.0, 4.0, 7.0, 7.0, 10.0, 8.0, 8.0, 6.0, 3.0, 8.0, 4.0, 5.0, 7.0, 8.0,
            6.0, 6.0, 4.0, 12.0, 9.0, 8.0, 8.0, 8.0, 6.0, 7.0, 4.0, 4.0, 5.0, 7.0,
        ];
        let coordinates = array(&x, &[81]);
        let values = array(&y, &[81]);
        let degrees = int_array(&[0, 1, 2, 3, 4], &[5]);
        let mut fitted = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyfit_degrees(coordinates, values, degrees, &mut fitted),
            OK
        );
        let expected = [
            -3.14526520e2,
            9.95368241,
            -1.13822012e-1,
            5.70886914e-4,
            -1.06123820e-6,
        ];
        for (actual, expected) in result_values(fitted).iter().zip(expected) {
            assert!((actual.re - expected).abs() <= 1.0e-6);
            assert!(actual.im.abs() <= 1.0e-12);
        }
        dispose(&[fitted, degrees, values, coordinates]);
    }
}

#[test]
fn numpy_legacy_polynomial_numerical_contracts_match_typed_operations() {
    unsafe {
        let coefficients = complex_array(
            &[
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 2.0),
                Complex64::new(0.0, 3.0),
            ],
            &[3],
        );
        let constants = array(&[], &[0]);
        let mut derivative = std::ptr::null_mut();
        let mut integral = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyder_axis(coefficients, 1, 1.0, 0, &mut derivative),
            OK
        );
        assert_eq!(
            terlan_ndarray_v1_polyint_axis(coefficients, 1, constants, 0.0, 1.0, 0, &mut integral,),
            OK
        );
        close(
            &result_values(derivative),
            &[Complex64::new(0.0, 2.0), Complex64::new(0.0, 6.0)],
        );
        close(
            &result_values(integral),
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 1.0),
            ],
        );

        let real_coefficients = array(&[1.0, 2.0, 3.0], &[3]);
        let staged_constants = array(&[9.0, 7.0, 6.0], &[3]);
        let mut staged = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyint_axis(
                real_coefficients,
                3,
                staged_constants,
                0.0,
                1.0,
                0,
                &mut staged,
            ),
            OK
        );
        close(
            &result_values(staged),
            &real(&[6.0, 7.0, 4.5, 1.0 / 6.0, 1.0 / 12.0, 1.0 / 20.0]),
        );

        let repeated_zero_coefficients = array(&[0.0, 0.0, 1.0], &[3]);
        let mut repeated_zero_roots = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyroots(repeated_zero_coefficients, &mut repeated_zero_roots),
            OK
        );
        close(
            &result_values(repeated_zero_roots),
            &[Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        );

        let dividend = complex_array(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            &[4],
        );
        let divisor = complex_array(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(-2.0, -1.0),
                Complex64::new(1.0, 2.0),
                Complex64::new(0.0, -1.0),
            ],
            &[4],
        );
        let mut outputs = std::ptr::null_mut();
        let mut output_length = 0;
        assert_eq!(
            terlan_ndarray_v1_polydiv_all(dividend, divisor, &mut outputs, &mut output_length,),
            OK
        );
        let quotient_remainder = returned_arrays(outputs, output_length);
        assert_eq!(
            super::super::native_support::dtype(quotient_remainder[0]).unwrap(),
            13
        );
        assert_eq!(
            super::super::native_support::dtype(quotient_remainder[1]).unwrap(),
            13
        );
        let mut product = std::ptr::null_mut();
        let mut reconstructed = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polymul(quotient_remainder[0], divisor, &mut product,),
            OK
        );
        assert_eq!(
            terlan_ndarray_v1_polyadd(product, quotient_remainder[1], &mut reconstructed),
            OK
        );
        close(&result_values(reconstructed), &result_values(dividend));

        dispose(&[reconstructed, product]);
        dispose(&quotient_remainder);
        dispose(&[
            divisor,
            dividend,
            repeated_zero_roots,
            repeated_zero_coefficients,
            staged,
            staged_constants,
            real_coefficients,
            integral,
            derivative,
            constants,
            coefficients,
        ]);
    }
}

#[test]
fn numpy_polynomial_completion_rejects_invalid_controls() {
    unsafe {
        let values = array(&[1.0, 2.0], &[2]);
        let empty = array(&[], &[0]);
        let float_degrees = array(&[0.0, 1.0], &[2]);
        let duplicate_degrees = int_array(&[0, 0], &[2]);
        let constants = array(&[1.0, 2.0], &[2]);
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyder_axis(values, -1, 1.0, 0, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_polyder_axis(values, 1, 1.0, 2, &mut output),
            super::super::native_support::AXIS_OUT_OF_RANGE
        );
        assert_eq!(
            terlan_ndarray_v1_polyint_axis(values, 1, constants, 0.0, 1.0, 0, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_polyfit_degrees(values, values, empty, &mut output),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_polyfit_degrees(values, values, float_degrees, &mut output),
            UNSUPPORTED_DTYPE
        );
        assert_eq!(
            terlan_ndarray_v1_polyfit_degrees(values, values, duplicate_degrees, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_polyvalfromroots_tensor(values, values, 2, &mut output),
            INVALID_ARGUMENT
        );
        let mut outputs = std::ptr::null_mut();
        let mut output_length = 0;
        assert_eq!(
            terlan_ndarray_v1_polyfit_covariance(
                values,
                values,
                1,
                1,
                &mut outputs,
                &mut output_length,
            ),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_polyfit_covariance(
                values,
                values,
                1,
                2,
                &mut outputs,
                &mut output_length,
            ),
            INVALID_ARGUMENT
        );
        assert!(outputs.is_null());
        assert_eq!(output_length, 0);
        assert!(output.is_null());
        dispose(&[constants, duplicate_degrees, float_degrees, empty, values]);
    }
}
