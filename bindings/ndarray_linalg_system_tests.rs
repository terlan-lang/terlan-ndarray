use super::*;
use crate::package_extension::native_support::{dtype, numeric, owned_numeric_as, shape};

/// Creates one owned numeric array with an explicit dtype.
unsafe fn array(values: &[c64], dimensions: &[i64], dtype_code: i32) -> Array {
    unsafe { owned_numeric_as(values, dimensions, dtype_code).unwrap() }
}

/// Reads one owned array as canonical complex values.
unsafe fn values(array: Array) -> Vec<c64> {
    unsafe { numeric(array, false).unwrap().values }
}

/// Converts and frees one transferred handle vector.
unsafe fn arrays(identities: *mut u64, length: usize) -> Vec<Array> {
    let output = unsafe { std::slice::from_raw_parts(identities, length) }
        .iter()
        .map(|&identity| identity as usize as Array)
        .collect();
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(identities) };
    output
}

/// Disposes every live owned test array exactly once.
unsafe fn dispose(arrays: &[Array]) {
    for &array in arrays {
        if !array.is_null() {
            assert_eq!(unsafe { crate::ffi::terlan_ndarray_v1_delete(array) }, OK);
        }
    }
}

/// Asserts one numerical value within a stable decomposition tolerance.
fn close(actual: c64, expected: c64) {
    assert!(
        (actual - expected).norm() < 1.0e-5,
        "{actual:?} != {expected:?}"
    );
}

#[test]
fn determinant_and_inverse_cover_stacks_dtypes_empty_and_singular_inputs() {
    unsafe {
        let stacked = array(
            &[
                c64::new(2.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(3.0, 0.0),
                c64::new(4.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(5.0, 0.0),
            ],
            &[2, 2, 2],
            11,
        );
        let mut determinants = std::ptr::null_mut();
        assert_eq!(determinant(stacked, &mut determinants), OK);
        assert_eq!(shape(determinants).unwrap(), [2]);
        assert_eq!(dtype(determinants).unwrap(), 11);
        assert_eq!(
            values(determinants),
            [c64::new(6.0, 0.0), c64::new(20.0, 0.0)]
        );

        let mut inverses = std::ptr::null_mut();
        assert_eq!(inverse(stacked, &mut inverses), OK);
        assert_eq!(shape(inverses).unwrap(), [2, 2, 2]);
        let inverse_values = values(inverses);
        for (actual, expected) in
            inverse_values
                .iter()
                .zip([0.5, 0.0, 0.0, 1.0 / 3.0, 0.25, 0.0, 0.0, 0.2])
        {
            close(*actual, c64::new(expected, 0.0));
        }

        for dtype_code in [11, 3, 12, 13] {
            let typed = array(
                &[
                    c64::new(2.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(3.0, 0.0),
                ],
                &[2, 2],
                dtype_code,
            );
            let mut typed_det = std::ptr::null_mut();
            let mut typed_inverse = std::ptr::null_mut();
            assert_eq!(determinant(typed, &mut typed_det), OK);
            assert_eq!(inverse(typed, &mut typed_inverse), OK);
            assert_eq!(dtype(typed_det).unwrap(), dtype_code);
            assert_eq!(dtype(typed_inverse).unwrap(), dtype_code);
            close(values(typed_det)[0], c64::new(6.0, 0.0));
            dispose(&[typed, typed_det, typed_inverse]);
        }

        let integers = array(
            &[
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
            ],
            &[2, 2],
            7,
        );
        let mut integer_inverse = std::ptr::null_mut();
        assert_eq!(inverse(integers, &mut integer_inverse), OK);
        assert_eq!(dtype(integer_inverse).unwrap(), 3);

        let empty = array(&[], &[3, 0, 0], 13);
        let mut empty_det = std::ptr::null_mut();
        let mut empty_inverse = std::ptr::null_mut();
        assert_eq!(determinant(empty, &mut empty_det), OK);
        assert_eq!(values(empty_det), vec![c64::new(1.0, 0.0); 3]);
        assert_eq!(inverse(empty, &mut empty_inverse), OK);
        assert_eq!(shape(empty_inverse).unwrap(), [3, 0, 0]);

        let singular = array(
            &[
                c64::new(1.0, 0.0),
                c64::new(2.0, 0.0),
                c64::new(2.0, 0.0),
                c64::new(4.0, 0.0),
            ],
            &[2, 2],
            3,
        );
        let mut rejected = std::ptr::null_mut();
        assert_eq!(inverse(singular, &mut rejected), DOMAIN_ERROR);
        assert!(rejected.is_null());
        let mut zero_det = std::ptr::null_mut();
        assert_eq!(determinant(singular, &mut zero_det), OK);
        close(values(zero_det)[0], c64::new(0.0, 0.0));
        dispose(&[
            stacked,
            determinants,
            inverses,
            integers,
            integer_inverse,
            empty,
            empty_det,
            empty_inverse,
            singular,
            zero_det,
        ]);
    }
}

#[test]
fn stacked_inverse_reports_an_early_singular_batch_atomically() {
    unsafe {
        let source = array(
            &[
                c64::new(1.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(2.0, 0.0),
                c64::new(3.0, 0.0),
                c64::new(4.0, 0.0),
            ],
            &[2, 2, 2],
            3,
        );
        let mut output = std::ptr::null_mut();
        assert_eq!(inverse(source, &mut output), DOMAIN_ERROR);
        assert!(output.is_null());
        assert_eq!(shape(source).unwrap(), [2, 2, 2]);
        dispose(&[source]);
    }
}

#[test]
fn solve_broadcasts_vector_and_matrix_rhs_and_preserves_promoted_dtype() {
    unsafe {
        let coefficients = array(
            &[
                c64::new(2.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(4.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(5.0, 0.0),
            ],
            &[2, 2, 2],
            11,
        );
        let matrix_rhs = array(&[c64::new(4.0, 0.0), c64::new(20.0, 0.0)], &[1, 2, 1], 11);
        let mut matrix_result = std::ptr::null_mut();
        assert_eq!(solve(coefficients, matrix_rhs, &mut matrix_result), OK);
        assert_eq!(shape(matrix_result).unwrap(), [2, 2, 1]);
        assert_eq!(dtype(matrix_result).unwrap(), 11);
        assert_eq!(
            values(matrix_result),
            [
                c64::new(2.0, 0.0),
                c64::new(5.0, 0.0),
                c64::new(4.0, 0.0),
                c64::new(4.0, 0.0),
            ]
        );

        let vector_rhs = array(
            &[
                c64::new(2.0, 0.0),
                c64::new(8.0, 0.0),
                c64::new(3.0, 0.0),
                c64::new(15.0, 0.0),
            ],
            &[2, 2],
            3,
        );
        let mut vector_result = std::ptr::null_mut();
        assert_eq!(solve(coefficients, vector_rhs, &mut vector_result), OK);
        assert_eq!(shape(vector_result).unwrap(), [2, 2]);
        assert_eq!(dtype(vector_result).unwrap(), 3);
        assert_eq!(
            values(vector_result),
            [
                c64::new(1.0, 0.0),
                c64::new(2.0, 0.0),
                c64::new(3.0, 0.0),
                c64::new(3.0, 0.0),
            ]
        );

        for dtype_code in [11, 3, 12, 13] {
            let identity = array(
                &[
                    c64::new(1.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(1.0, 0.0),
                ],
                &[2, 2],
                dtype_code,
            );
            let typed_rhs = array(&[c64::new(2.0, 0.0), c64::new(3.0, 0.0)], &[2], dtype_code);
            let mut typed_result = std::ptr::null_mut();
            assert_eq!(solve(identity, typed_rhs, &mut typed_result), OK);
            assert_eq!(dtype(typed_result).unwrap(), dtype_code);
            assert_eq!(values(typed_result), values(typed_rhs));
            dispose(&[identity, typed_rhs, typed_result]);
        }

        let empty_coefficients = array(&[], &[2, 0, 0], 12);
        let empty_rhs = array(&[], &[1, 0, 3], 12);
        let mut empty_result = std::ptr::null_mut();
        assert_eq!(solve(empty_coefficients, empty_rhs, &mut empty_result), OK);
        assert_eq!(shape(empty_result).unwrap(), [2, 0, 3]);
        assert_eq!(dtype(empty_result).unwrap(), 12);

        let incompatible = array(&[c64::new(1.0, 0.0); 6], &[1, 3, 2], 3);
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            solve(coefficients, incompatible, &mut rejected),
            SHAPE_MISMATCH
        );
        assert!(rejected.is_null());
        dispose(&[
            coefficients,
            matrix_rhs,
            matrix_result,
            vector_rhs,
            vector_result,
            empty_coefficients,
            empty_rhs,
            empty_result,
            incompatible,
        ]);
    }
}

#[test]
fn condition_handles_stacked_singular_and_rectangular_matrices() {
    unsafe {
        let stacked = array(
            &[
                c64::new(2.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.5, 0.0),
                c64::new(1.0, 0.0),
                c64::new(2.0, 0.0),
                c64::new(2.0, 0.0),
                c64::new(4.0, 0.0),
            ],
            &[2, 2, 2],
            3,
        );
        let mut output = std::ptr::null_mut();
        assert_eq!(condition(stacked, 2, &mut output), OK);
        assert_eq!(shape(output).unwrap(), [2]);
        let condition_values = values(output);
        close(condition_values[0], c64::new(4.0, 0.0));
        assert!(condition_values[1].re.is_infinite());

        let rectangular = array(
            &[
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
            ],
            &[3, 2],
            3,
        );
        let mut reciprocal = std::ptr::null_mut();
        assert_eq!(condition(rectangular, -2, &mut reciprocal), OK);
        close(values(reciprocal)[0], c64::new(1.0, 0.0));
        dispose(&[stacked, output, rectangular, reciprocal]);
    }
}

#[test]
fn eig_and_eigvals_cover_stacks_real_complex_dtypes_and_empty_matrices() {
    unsafe {
        let stacked = array(
            &[
                c64::new(0.0, 0.0),
                c64::new(-1.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(2.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(3.0, 0.0),
            ],
            &[2, 2, 2],
            11,
        );
        let mut identities = std::ptr::null_mut();
        let mut length = 0usize;
        assert_eq!(eig(stacked, &mut identities, &mut length), OK);
        assert_eq!(length, 2);
        let outputs = arrays(identities, length);
        assert_eq!(shape(outputs[0]).unwrap(), [2, 2]);
        assert_eq!(shape(outputs[1]).unwrap(), [2, 2, 2]);
        assert_eq!(dtype(outputs[0]).unwrap(), 12);
        assert_eq!(dtype(outputs[1]).unwrap(), 12);
        assert!(values(outputs[0]).iter().any(|value| value.im.abs() > 0.9));

        let diagonal = array(
            &[
                c64::new(2.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(3.0, 0.0),
            ],
            &[2, 2],
            11,
        );
        let mut real_values = std::ptr::null_mut();
        assert_eq!(eigvals(diagonal, &mut real_values), OK);
        assert_eq!(dtype(real_values).unwrap(), 11);

        for dtype_code in [11, 3, 12, 13] {
            let typed = array(
                &[
                    c64::new(2.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(3.0, 0.0),
                ],
                &[2, 2],
                dtype_code,
            );
            let mut typed_identities = std::ptr::null_mut();
            let mut typed_length = 0usize;
            assert_eq!(eig(typed, &mut typed_identities, &mut typed_length), OK);
            let typed_outputs = arrays(typed_identities, typed_length);
            let mut typed_values = std::ptr::null_mut();
            assert_eq!(eigvals(typed, &mut typed_values), OK);
            assert_eq!(dtype(typed_outputs[0]).unwrap(), dtype_code);
            assert_eq!(dtype(typed_outputs[1]).unwrap(), dtype_code);
            assert_eq!(dtype(typed_values).unwrap(), dtype_code);
            dispose(&[typed, typed_outputs[0], typed_outputs[1], typed_values]);
        }

        let empty = array(&[], &[3, 0, 0], 13);
        let mut empty_values = std::ptr::null_mut();
        assert_eq!(eigvals(empty, &mut empty_values), OK);
        assert_eq!(shape(empty_values).unwrap(), [3, 0]);
        assert_eq!(dtype(empty_values).unwrap(), 13);
        dispose(&[
            stacked,
            outputs[0],
            outputs[1],
            diagonal,
            real_values,
            empty,
            empty_values,
        ]);
    }
}

#[test]
fn eigen_build_regressions_match_numpy_reference_values() {
    unsafe {
        let source_values = (0..13 * 13)
            .map(|index| c64::new((index % 17) as f64, 0.0))
            .collect::<Vec<_>>();
        let source = array(&source_values, &[13, 13], 3);
        let mut identities = std::ptr::null_mut();
        let mut length = 0usize;
        assert_eq!(eig(source, &mut identities, &mut length), OK);
        let factors = arrays(identities, length);
        let actual = values(factors[0]);
        let mut expected = vec![
            c64::new(103.221168, 0.0),
            c64::new(-19.1843603, 0.0),
            c64::new(-0.604004526, 15.84422474),
            c64::new(-0.604004526, -15.84422474),
            c64::new(-11.3692929, 0.0),
            c64::new(-0.657612485, 10.41755503),
            c64::new(-0.657612485, -10.41755503),
            c64::new(18.2126812, 0.0),
            c64::new(10.6011014, 0.0),
            c64::new(7.80732773, 0.0),
            c64::new(-0.765390898, 0.0),
            c64::new(1.51971555e-15, 0.0),
            c64::new(-1.51308713e-15, 0.0),
        ];
        for observed in actual {
            let (index, distance) = expected
                .iter()
                .enumerate()
                .map(|(index, reference)| (index, (observed - reference).norm()))
                .min_by(|left, right| left.1.total_cmp(&right.1))
                .unwrap();
            assert!(
                distance < 1.0e-4,
                "no NumPy reference eigenvalue matches {observed:?}",
            );
            expected.remove(index);
        }
        assert!(expected.is_empty());

        let covariance = array(
            &[
                c64::new(77.70273908, 0.0),
                c64::new(3.51489954, 0.0),
                c64::new(15.64602427, 0.0),
                c64::new(3.51489954, 0.0),
                c64::new(88.97013878, 0.0),
                c64::new(-1.07431931, 0.0),
                c64::new(15.64602427, 0.0),
                c64::new(-1.07431931, 0.0),
                c64::new(98.18223512, 0.0),
            ],
            &[3, 3],
            3,
        );
        let mut eigen = std::ptr::null_mut();
        assert_eq!(eigvals(covariance, &mut eigen), OK);
        let mut observed = values(eigen);
        observed.sort_by(|left, right| left.re.total_cmp(&right.re));
        for (actual, expected) in observed
            .iter()
            .zip([68.60568999, 89.57756725, 106.67185574])
        {
            assert!((actual.re - expected).abs() < 1.0e-5);
        }
        dispose(&[source, factors[0], factors[1], covariance, eigen]);
    }
}
