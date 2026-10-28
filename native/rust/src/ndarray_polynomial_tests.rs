use super::*;

const OK: i32 = 0;

/// Creates one owned Float64 fixture array.
unsafe fn array(values: &[f64], shape: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, shape, 3).unwrap() }
}

/// Creates one owned Complex128 fixture array.
unsafe fn complex_array(values: &[Complex64], shape: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_complex(values, shape, 13).unwrap() }
}

/// Reads one fixture array as logical complex values.
unsafe fn complex_values(array: Array) -> Vec<Complex64> {
    unsafe {
        super::super::native_support::numeric(array, false)
            .unwrap()
            .values
    }
}

/// Reads one fixture array as logical Float64 values.
unsafe fn values(array: Array) -> Vec<f64> {
    unsafe { super::super::native_support::floats(array, false).unwrap() }
}

/// Deletes all supplied fixture arrays.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { super::super::native_support::delete_arrays(arrays) }
}

/// Asserts approximate scalar-vector equality.
fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-8, "{actual} != {expected}");
    }
}

/// Asserts approximate complex-vector equality.
fn complex_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).norm() < 1.0e-8,
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn evaluation_roots_and_fitting_match_power_basis_contract() {
    unsafe {
        let coefficients = array(&[2.0, -3.0, 1.0], &[3]);
        let coordinates = array(&[0.0, 1.0, 2.0, 3.0], &[2, 2]);
        let mut evaluated = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyval(coordinates, coefficients, &mut evaluated),
            OK
        );
        close(&values(evaluated), &[2.0, 0.0, 0.0, 2.0]);

        let mut roots = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_polyroots(coefficients, &mut roots), OK);
        let mut root_values = Vec::new();
        let mut reals = std::ptr::null_mut();
        let mut length = 0;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_complex_reals(roots, &mut reals, &mut length),
            OK
        );
        root_values.extend_from_slice(std::slice::from_raw_parts(reals, length));
        crate::ffi::terlan_ndarray_v1_delete_float_values(reals);
        root_values.sort_by(f64::total_cmp);
        close(&root_values, &[1.0, 2.0]);

        let x = array(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let y = array(&[3.0, 0.0, 1.0, 6.0, 15.0], &[5]);
        let mut fitted = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_polyfit(x, y, 2, &mut fitted), OK);
        close(&values(fitted), &[1.0, 3.0, 2.0]);
        dispose(&[fitted, y, x, roots, evaluated, coordinates, coefficients]);
    }
}

#[test]
fn arithmetic_derivative_integral_division_and_companion_are_consistent() {
    unsafe {
        let lhs = array(&[2.0, -3.0, 1.0], &[3]);
        let rhs = array(&[-1.0, 1.0], &[2]);
        let mut added = std::ptr::null_mut();
        let mut subtracted = std::ptr::null_mut();
        let mut multiplied = std::ptr::null_mut();
        let mut derivative = std::ptr::null_mut();
        let mut integral = std::ptr::null_mut();
        let mut companion = std::ptr::null_mut();
        let mut converted = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_polyadd(lhs, rhs, &mut added), OK);
        assert_eq!(terlan_ndarray_v1_polysub(lhs, rhs, &mut subtracted), OK);
        assert_eq!(terlan_ndarray_v1_polymul(rhs, rhs, &mut multiplied), OK);
        assert_eq!(terlan_ndarray_v1_polyder(lhs, 1, &mut derivative), OK);
        assert_eq!(
            terlan_ndarray_v1_polyint(derivative, 1, 2.0, &mut integral),
            OK
        );
        assert_eq!(terlan_ndarray_v1_polycompanion(lhs, &mut companion), OK);
        assert_eq!(
            terlan_ndarray_v1_polyconvert(lhs, 0, 1, -1.0, 1.0, -1.0, 1.0, &mut converted),
            OK
        );
        close(&values(added), &[1.0, -2.0, 1.0]);
        close(&values(subtracted), &[3.0, -4.0, 1.0]);
        close(&values(multiplied), &[1.0, -2.0, 1.0]);
        close(&values(derivative), &[-3.0, 2.0]);
        close(&values(integral), &[2.0, -3.0, 1.0]);
        close(&values(companion), &[0.0, -2.0, 1.0, 3.0]);
        close(&values(converted), &[2.5, -3.0, 0.5]);

        let mut outputs = std::ptr::null_mut();
        let mut output_length = 0;
        assert_eq!(
            terlan_ndarray_v1_polydiv_all(lhs, rhs, &mut outputs, &mut output_length),
            OK
        );
        assert_eq!(output_length, 2);
        let handles = std::slice::from_raw_parts(outputs, output_length);
        close(&values(handles[0] as usize as Array), &[-2.0, 1.0]);
        close(&values(handles[1] as usize as Array), &[0.0]);
        for &handle in handles {
            dispose(&[handle as usize as Array]);
        }
        crate::ffi::terlan_ndarray_v1_delete_handle_values(outputs);
        dispose(&[
            converted, companion, integral, derivative, multiplied, subtracted, added, rhs, lhs,
        ]);
    }
}

#[test]
fn complex_polynomial_paths_preserve_values_and_return_complex_roots() {
    unsafe {
        let coefficients = complex_array(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            &[3],
        );
        let coordinates =
            complex_array(&[Complex64::new(0.0, 1.0), Complex64::new(1.0, 1.0)], &[2]);
        let mut evaluated = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyval(coordinates, coefficients, &mut evaluated),
            OK
        );
        complex_close(
            &complex_values(evaluated),
            &[Complex64::new(0.0, 0.0), Complex64::new(1.0, 2.0)],
        );

        let mut root_array = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyroots(coefficients, &mut root_array),
            OK
        );
        let mut roots = complex_values(root_array);
        roots.sort_by(|lhs, rhs| lhs.im.total_cmp(&rhs.im));
        complex_close(
            &roots,
            &[Complex64::new(0.0, -1.0), Complex64::new(0.0, 1.0)],
        );

        let x = array(&[0.0, 1.0, 2.0], &[3]);
        let y = complex_array(
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(3.0, 2.0),
                Complex64::new(5.0, 3.0),
            ],
            &[3],
        );
        let mut fitted = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_polyfit(x, y, 1, &mut fitted), OK);
        complex_close(
            &complex_values(fitted),
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, 1.0)],
        );
        dispose(&[
            fitted,
            y,
            x,
            root_array,
            evaluated,
            coordinates,
            coefficients,
        ]);
    }
}

#[test]
fn polynomial_operations_reject_degenerate_and_rank_deficient_inputs() {
    unsafe {
        let scalar = array(&[1.0], &[]);
        let constant = array(&[2.0], &[1]);
        let zero = array(&[0.0, 0.0], &[2]);
        let duplicate_x = array(&[1.0, 1.0, 1.0], &[3]);
        let y = array(&[1.0, 2.0, 3.0], &[3]);
        let mut output = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_polyroots(constant, &mut output), OK);
        assert_eq!(super::super::native_support::shape(output).unwrap(), [0]);
        dispose(&[output]);
        output = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_polyroots(zero, &mut output), OK);
        assert_eq!(super::super::native_support::shape(output).unwrap(), [0]);
        dispose(&[output]);
        output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyval(y, scalar, &mut output),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_polyfit(duplicate_x, y, 2, &mut output),
            DOMAIN_ERROR
        );
        assert_eq!(
            terlan_ndarray_v1_polyder(constant, -1, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_polyconvert(constant, 9, 0, -1.0, 1.0, -1.0, 1.0, &mut output),
            INVALID_ARGUMENT
        );
        let mut outputs = std::ptr::null_mut();
        let mut output_length = 0;
        assert_eq!(
            terlan_ndarray_v1_polydiv_all(y, zero, &mut outputs, &mut output_length),
            DOMAIN_ERROR
        );
        assert!(output.is_null());
        assert!(outputs.is_null());
        dispose(&[y, duplicate_x, zero, constant, scalar]);
    }
}
