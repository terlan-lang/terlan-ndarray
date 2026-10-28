use super::*;

type Array = *mut crate::ffi::TerlanNdArray;

unsafe fn array(values: &[f64], shape: &[i64]) -> Array {
    let mut result = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            crate::ffi::terlan_ndarray_v1_from_floats(
                values.as_ptr(),
                values.len() as i64,
                shape.as_ptr(),
                shape.len() as i64,
                &mut result,
            )
        },
        OK
    );
    result
}

unsafe fn dispose(arrays: &[Array]) {
    for &array in arrays {
        if !array.is_null() {
            assert_eq!(unsafe { crate::ffi::terlan_ndarray_v1_delete(array) }, OK);
        }
    }
}

unsafe fn result_arrays(values: *mut u64, length: usize) -> Vec<Array> {
    assert!(!values.is_null());
    let arrays = unsafe { std::slice::from_raw_parts(values, length) }
        .iter()
        .map(|identity| *identity as usize as Array)
        .collect();
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(values) };
    arrays
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-9,
        "{actual} != {expected}"
    );
}

#[test]
fn decomposition_core_solves_and_rejects_singular_or_non_finite_inputs() {
    unsafe {
        let matrix = array(&[4.0, 7.0, 2.0, 6.0], &[2, 2]);
        let rhs = array(&[1.0, 0.0], &[2]);
        let mut determinant = 0.0;
        assert_eq!(terlan_ndarray_v1_determinant(matrix, &mut determinant), OK);
        close(determinant, 10.0);
        let mut solution = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_solve(matrix, rhs, &mut solution), OK);
        for (actual, expected) in array_floats(solution).unwrap().iter().zip([0.6, -0.2]) {
            close(*actual, expected);
        }
        let mut inverse = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_inverse(matrix, &mut inverse), OK);
        for (actual, expected) in array_floats(inverse)
            .unwrap()
            .iter()
            .zip([0.6, -0.7, -0.2, 0.4])
        {
            close(*actual, expected);
        }
        let singular = array(&[1.0, 2.0, 2.0, 4.0], &[2, 2]);
        let non_finite = array(&[f64::NAN, 0.0, 0.0, 1.0], &[2, 2]);
        let mut failed = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_solve(singular, rhs, &mut failed),
            DOMAIN_ERROR
        );
        assert_eq!(
            terlan_ndarray_v1_inverse(non_finite, &mut failed),
            DOMAIN_ERROR
        );
        assert!(failed.is_null());
        dispose(&[matrix, rhs, solution, inverse, singular, non_finite]);
    }
}

#[test]
fn norms_rank_and_least_squares_match_reference_values() {
    unsafe {
        let matrix = array(&[1.0, -2.0, 3.0, 4.0], &[2, 2]);
        let vector = array(&[3.0, -4.0], &[2]);
        for (operation, expected) in [
            (
                terlan_ndarray_v1_norm_l1 as unsafe extern "C" fn(Array, *mut f64) -> i32,
                6.0,
            ),
            (terlan_ndarray_v1_norm_max, 7.0),
            (terlan_ndarray_v1_frobenius_norm, 30.0_f64.sqrt()),
            (terlan_ndarray_v1_nuclear_norm, 50.0_f64.sqrt()),
        ] {
            let mut value = 0.0;
            assert_eq!(operation(matrix, &mut value), OK);
            close(value, expected);
        }
        let mut value = 0.0;
        assert_eq!(terlan_ndarray_v1_norm_l2(vector, &mut value), OK);
        close(value, 5.0);
        let rank_deficient = array(&[1.0, 2.0, 2.0, 4.0], &[2, 2]);
        let mut rank = -1;
        assert_eq!(
            terlan_ndarray_v1_matrix_rank(rank_deficient, 0.0, &mut rank),
            OK
        );
        assert_eq!(rank, 1);
        let coefficients = array(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
        let rhs = array(&[1.0, 2.0, 2.0], &[3]);
        let mut solution = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_least_squares(coefficients, rhs, 0.0, &mut solution),
            OK
        );
        for (actual, expected) in array_floats(solution)
            .unwrap()
            .iter()
            .zip([2.0 / 3.0, 5.0 / 3.0])
        {
            close(*actual, expected);
        }
        assert_eq!(
            terlan_ndarray_v1_matrix_rank(matrix, -1.0, &mut rank),
            DOMAIN_ERROR
        );
        assert_eq!(
            terlan_ndarray_v1_nuclear_norm(vector, &mut value),
            SHAPE_MISMATCH
        );
        dispose(&[matrix, vector, rank_deficient, coefficients, rhs, solution]);
    }
}

#[test]
fn cholesky_qr_svd_and_eigen_transfer_owned_factors() {
    unsafe {
        let positive = array(&[4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let mut cholesky = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_cholesky(positive, &mut cholesky), OK);
        for (actual, expected) in
            array_floats(cholesky)
                .unwrap()
                .iter()
                .zip([2.0, 0.0, 1.0, 2.0_f64.sqrt()])
        {
            close(*actual, expected);
        }

        let tall = array(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
        let mut identities = std::ptr::null_mut();
        let mut length = 0;
        assert_eq!(
            terlan_ndarray_v1_qr(tall, 1, &mut identities, &mut length),
            OK
        );
        assert_eq!(length, 2);
        let qr = result_arrays(identities, length);
        assert_eq!(array_shape(qr[0]).unwrap(), [3, 2]);
        assert_eq!(array_shape(qr[1]).unwrap(), [2, 2]);

        identities = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_svd(tall, 0, &mut identities, &mut length),
            OK
        );
        assert_eq!(length, 3);
        let svd = result_arrays(identities, length);
        assert_eq!(array_shape(svd[0]).unwrap(), [3, 2]);
        assert_eq!(array_shape(svd[1]).unwrap(), [2]);
        assert_eq!(array_shape(svd[2]).unwrap(), [2, 2]);
        let singular = array_floats(svd[1]).unwrap();
        assert!(singular[0] >= singular[1] && singular[1] >= 0.0);

        let symmetric = array(&[2.0, 1.0, 1.0, 2.0], &[2, 2]);
        identities = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_symmetric_eigen(symmetric, 0.0, &mut identities, &mut length),
            OK
        );
        let eigen = result_arrays(identities, length);
        assert_eq!(length, 2);
        for (actual, expected) in array_floats(eigen[0]).unwrap().iter().zip([1.0, 3.0]) {
            close(*actual, expected);
        }
        let not_positive = array(&[1.0, 2.0, 2.0, 1.0], &[2, 2]);
        let not_symmetric = array(&[1.0, 2.0, 0.0, 1.0], &[2, 2]);
        let mut failed = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_cholesky(not_positive, &mut failed),
            DOMAIN_ERROR
        );
        assert_eq!(
            terlan_ndarray_v1_symmetric_eigen(not_symmetric, 0.0, &mut identities, &mut length),
            OK
        );
        let selected_triangle = result_arrays(identities, length);
        assert_eq!(array_floats(selected_triangle[0]).unwrap(), [1.0, 1.0]);
        dispose(&[
            positive,
            cholesky,
            tall,
            symmetric,
            not_positive,
            not_symmetric,
        ]);
        dispose(&qr);
        dispose(&svd);
        dispose(&eigen);
        dispose(&selected_triangle);
    }
}

#[test]
fn decompositions_reject_invalid_flags_shapes_outputs_and_tolerances() {
    unsafe {
        let vector = array(&[1.0, 2.0], &[2]);
        let matrix = array(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let bad_rhs = array(&[1.0, 2.0, 3.0], &[3]);
        let mut output = std::ptr::null_mut();
        let mut array_output = std::ptr::null_mut();
        let mut length = 99;
        assert_eq!(
            terlan_ndarray_v1_qr(matrix, 2, &mut output, &mut length),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_svd(matrix, -1, &mut output, &mut length),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_qr(vector, 1, &mut output, &mut length),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_least_squares(matrix, bad_rhs, 0.0, &mut array_output),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_least_squares(matrix, vector, f64::NAN, &mut array_output),
            DOMAIN_ERROR
        );
        assert_eq!(
            terlan_ndarray_v1_qr(matrix, 1, std::ptr::null_mut(), &mut length),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_symmetric_eigen(matrix, 0.0, &mut output, std::ptr::null_mut()),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert!(array_output.is_null());
        dispose(&[vector, matrix, bad_rhs]);
    }
}

#[test]
fn decomposition_conformance_covers_empty_ill_conditioned_and_strided_inputs() {
    unsafe {
        let empty = array(&[], &[0, 0]);
        let mut scalar = -1.0;
        assert_eq!(terlan_ndarray_v1_determinant(empty, &mut scalar), OK);
        close(scalar, 1.0);
        assert_eq!(terlan_ndarray_v1_frobenius_norm(empty, &mut scalar), OK);
        close(scalar, 0.0);
        let mut rank = -1;
        assert_eq!(terlan_ndarray_v1_matrix_rank(empty, 0.0, &mut rank), OK);
        assert_eq!(rank, 0);

        let ill_conditioned = array(&[1.0, 0.0, 0.0, 1.0e-20], &[2, 2]);
        assert_eq!(
            terlan_ndarray_v1_matrix_rank(ill_conditioned, 0.0, &mut rank),
            OK
        );
        assert_eq!(rank, 1);

        let source = array(&[1.0, 2.0, 3.0, 5.0, 7.0, 11.0], &[2, 3]);
        let mut transposed = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_transpose(source, 0, 1, &mut transposed),
            OK
        );
        assert_eq!(
            terlan_ndarray_v1_frobenius_norm(transposed, &mut scalar),
            OK
        );
        close(scalar, 209.0_f64.sqrt());
        let mut identities = std::ptr::null_mut();
        let mut length = 0;
        assert_eq!(
            terlan_ndarray_v1_svd(empty, 0, &mut identities, &mut length),
            OK
        );
        let empty_svd = result_arrays(identities, length);
        assert_eq!(length, 3);
        assert_eq!(array_shape(empty_svd[0]).unwrap(), [0, 0]);
        assert_eq!(array_shape(empty_svd[1]).unwrap(), [0]);
        assert_eq!(array_shape(empty_svd[2]).unwrap(), [0, 0]);
        dispose(&[empty, ill_conditioned, source, transposed]);
        dispose(&empty_svd);
    }
}
