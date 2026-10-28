use super::*;
use crate::package_extension::native_support::{dtype, numeric, owned_numeric_as, shape, OK};

/// Creates one owned array with an explicit numerical dtype.
unsafe fn array(values: &[c64], dimensions: &[i64], dtype_code: i32) -> Array {
    unsafe { owned_numeric_as(values, dimensions, dtype_code).unwrap() }
}

/// Disposes every owned test array.
unsafe fn dispose(arrays: &[Array]) {
    for &array in arrays {
        if !array.is_null() {
            assert_eq!(unsafe { crate::ffi::terlan_ndarray_v1_delete(array) }, OK);
        }
    }
}

/// Converts and frees one transferred handle vector.
unsafe fn arrays(values: *mut u64, length: usize) -> Vec<Array> {
    let output = unsafe { std::slice::from_raw_parts(values, length) }
        .iter()
        .map(|&identity| identity as usize as Array)
        .collect();
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(values) };
    output
}

/// Reads one test array as canonical complex values.
unsafe fn values(array: Array) -> Vec<c64> {
    unsafe { numeric(array, false).unwrap().values }
}

/// Asserts one complex value within a stable numerical tolerance.
fn close(actual: c64, expected: c64) {
    assert!(
        (actual - expected).norm() < 1.0e-5,
        "{actual:?} != {expected:?}"
    );
}

/// Multiplies two row-major matrices for decomposition reconstruction checks.
fn multiply(
    left: &[c64],
    left_rows: usize,
    inner: usize,
    right: &[c64],
    right_columns: usize,
) -> Vec<c64> {
    let mut output = vec![c64::new(0.0, 0.0); left_rows * right_columns];
    for row in 0..left_rows {
        for column in 0..right_columns {
            for index in 0..inner {
                output[row * right_columns + column] +=
                    left[row * inner + index] * right[index * right_columns + column];
            }
        }
    }
    output
}

/// Builds repeated row-major identity matrices.
fn identities(count: usize, dimension: usize) -> Vec<c64> {
    let mut values = Vec::with_capacity(count * dimension * dimension);
    for _ in 0..count {
        append_identity(&mut values, dimension, dimension);
    }
    values
}

#[test]
fn qr_modes_cover_reduced_complete_r_raw_dtypes_and_empty_matrices() {
    unsafe {
        for (rows, columns) in [(2usize, 2usize), (3, 2), (2, 3)] {
            let source_values = (0..rows * columns)
                .map(|index| c64::new(index as f64 + 1.0, index as f64 * 0.125))
                .collect::<Vec<_>>();
            for dtype_code in [11, 3, 12, 13] {
                let values = if matches!(dtype_code, 12 | 13) {
                    source_values.clone()
                } else {
                    source_values
                        .iter()
                        .map(|value| c64::new(value.re, 0.0))
                        .collect()
                };
                let source = array(&values, &[rows as i64, columns as i64], dtype_code);
                let inner = rows.min(columns);
                for (mode, expected_length) in [(0i64, 2usize), (1, 2), (2, 1), (3, 2)] {
                    let mut identities = std::ptr::null_mut();
                    let mut length = 0usize;
                    assert_eq!(qr_mode(source, mode, &mut identities, &mut length), OK);
                    assert_eq!(length, expected_length);
                    let output = arrays(identities, length);
                    for &item in &output {
                        assert_eq!(dtype(item).unwrap(), dtype_code);
                    }
                    match mode {
                        0 => {
                            assert_eq!(shape(output[0]).unwrap(), [rows as i64, inner as i64]);
                            assert_eq!(shape(output[1]).unwrap(), [inner as i64, columns as i64]);
                        }
                        1 => {
                            assert_eq!(shape(output[0]).unwrap(), [rows as i64, rows as i64]);
                            assert_eq!(shape(output[1]).unwrap(), [rows as i64, columns as i64]);
                        }
                        2 => assert_eq!(shape(output[0]).unwrap(), [inner as i64, columns as i64]),
                        3 => {
                            assert_eq!(shape(output[0]).unwrap(), [columns as i64, rows as i64]);
                            assert_eq!(shape(output[1]).unwrap(), [inner as i64]);
                        }
                        _ => unreachable!(),
                    }
                    dispose(&output);
                }
                dispose(&[source]);
            }
        }

        for (rows, columns) in [(3usize, 0usize), (0, 3), (0, 0)] {
            let source = array(&[], &[rows as i64, columns as i64], 3);
            let inner = rows.min(columns);
            for mode in 0i64..=3 {
                let mut identities = std::ptr::null_mut();
                let mut length = 0usize;
                assert_eq!(qr_mode(source, mode, &mut identities, &mut length), OK);
                let output = arrays(identities, length);
                match mode {
                    0 => {
                        assert_eq!(shape(output[0]).unwrap(), [rows as i64, inner as i64]);
                        assert_eq!(shape(output[1]).unwrap(), [inner as i64, columns as i64]);
                    }
                    1 => {
                        assert_eq!(shape(output[0]).unwrap(), [rows as i64, rows as i64]);
                        assert_eq!(shape(output[1]).unwrap(), [rows as i64, columns as i64]);
                    }
                    2 => assert_eq!(shape(output[0]).unwrap(), [inner as i64, columns as i64]),
                    3 => {
                        assert_eq!(shape(output[0]).unwrap(), [columns as i64, rows as i64]);
                        assert_eq!(shape(output[1]).unwrap(), [inner as i64]);
                    }
                    _ => unreachable!(),
                }
                dispose(&output);
            }
            dispose(&[source]);
        }

        let source = array(&[c64::new(1.0, 0.0)], &[1, 1], 3);
        let mut rejected = std::ptr::null_mut();
        let mut rejected_length = 0usize;
        assert_eq!(
            qr_mode(source, 4, &mut rejected, &mut rejected_length),
            INVALID_ARGUMENT
        );
        assert!(rejected.is_null());
        assert_eq!(rejected_length, 0);
        dispose(&[source]);
    }
}

#[test]
fn cholesky_preserves_numpy_dtypes_batches_empty_shapes_and_selected_triangle() {
    unsafe {
        for dtype_code in [11, 3, 12, 13] {
            let complex = matches!(dtype_code, 12 | 13);
            let lower = if complex {
                c64::new(1.0, -2.0)
            } else {
                c64::new(1.0, 0.0)
            };
            let source = array(
                &[
                    c64::new(5.0, 0.0),
                    if complex {
                        c64::new(99.0, 99.0)
                    } else {
                        c64::new(99.0, 0.0)
                    },
                    lower,
                    c64::new(5.0, 0.0),
                ],
                &[2, 2],
                dtype_code,
            );
            let mut output = std::ptr::null_mut();
            assert_eq!(cholesky(source, &mut output), OK);
            assert_eq!(dtype(output).unwrap(), dtype_code);
            let result = values(output);
            close(result[0], c64::new(5.0f64.sqrt(), 0.0));
            close(result[1], c64::new(0.0, 0.0));
            close(result[2], lower / 5.0f64.sqrt());
            close(
                result[3],
                c64::new((5.0 - lower.norm_sqr() / 5.0).sqrt(), 0.0),
            );
            dispose(&[source, output]);
        }

        for (dimensions, batches, dimension) in [
            (vec![1, 1], 1, 1),
            (vec![2, 2], 1, 2),
            (vec![3, 3], 1, 3),
            (vec![50, 50], 1, 50),
            (vec![3, 10, 10], 3, 10),
        ] {
            for dtype_code in [11, 3, 12, 13] {
                let expected = identities(batches, dimension);
                let source = array(&expected, &dimensions, dtype_code);
                let mut output = std::ptr::null_mut();
                assert_eq!(cholesky(source, &mut output), OK);
                assert_eq!(shape(output).unwrap(), dimensions);
                assert_eq!(dtype(output).unwrap(), dtype_code);
                assert_eq!(values(output), expected);
                dispose(&[source, output]);
            }
        }

        let batched = array(
            &[
                c64::new(4.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(2.0, 0.0),
                c64::new(3.0, 0.0),
                c64::new(9.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(4.0, 0.0),
            ],
            &[2, 2, 2],
            3,
        );
        let mut result = std::ptr::null_mut();
        assert_eq!(cholesky(batched, &mut result), OK);
        assert_eq!(shape(result).unwrap(), [2, 2, 2]);
        close(values(result)[4], c64::new(3.0, 0.0));

        let empty_batches = array(&[], &[0, 1, 1], 2);
        let mut empty_batch_result = std::ptr::null_mut();
        assert_eq!(cholesky(empty_batches, &mut empty_batch_result), OK);
        assert_eq!(shape(empty_batch_result).unwrap(), [0, 1, 1]);
        assert_eq!(dtype(empty_batch_result).unwrap(), 3);

        let empty_matrix = array(&[], &[1, 0, 0], 12);
        let mut empty_matrix_result = std::ptr::null_mut();
        assert_eq!(cholesky(empty_matrix, &mut empty_matrix_result), OK);
        assert_eq!(shape(empty_matrix_result).unwrap(), [1, 0, 0]);
        assert_eq!(dtype(empty_matrix_result).unwrap(), 12);

        let float16 = array(&[c64::new(1.0, 0.0)], &[1, 1], 14);
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            cholesky(float16, &mut rejected),
            crate::package_extension::native_support::UNSUPPORTED_DTYPE
        );
        assert!(rejected.is_null());
        dispose(&[
            batched,
            result,
            empty_batches,
            empty_batch_result,
            empty_matrix,
            empty_matrix_result,
            float16,
        ]);
    }
}

#[test]
fn qr_preserves_numpy_shapes_dtypes_complex_values_batches_and_empty_identities() {
    unsafe {
        for dtype_code in [11, 3, 12, 13] {
            let source_values = if matches!(dtype_code, 12 | 13) {
                [
                    c64::new(1.0, 1.0),
                    c64::new(2.0, -1.0),
                    c64::new(3.0, 0.5),
                    c64::new(4.0, 2.0),
                    c64::new(5.0, -2.0),
                    c64::new(6.0, 1.0),
                ]
            } else {
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(|value| c64::new(value, 0.0))
            };
            let source = array(&source_values, &[3, 2], dtype_code);
            for reduced in [false, true] {
                let mut identities = std::ptr::null_mut();
                let mut length = 0;
                assert_eq!(qr(source, reduced, &mut identities, &mut length), OK);
                let factors = arrays(identities, length);
                assert_eq!(length, 2);
                assert_eq!(dtype(factors[0]).unwrap(), dtype_code);
                assert_eq!(dtype(factors[1]).unwrap(), dtype_code);
                let inner = if reduced { 2 } else { 3 };
                assert_eq!(shape(factors[0]).unwrap(), [3, inner as i64]);
                assert_eq!(shape(factors[1]).unwrap(), [inner as i64, 2]);
                let reconstructed = multiply(&values(factors[0]), 3, inner, &values(factors[1]), 2);
                for (actual, expected) in reconstructed.iter().zip(values(source)) {
                    close(*actual, expected);
                }
                dispose(&factors);
            }
            dispose(&[source]);
        }

        let empty_tall = array(&[], &[4, 0], 3);
        let mut identities = std::ptr::null_mut();
        let mut length = 0;
        assert_eq!(qr(empty_tall, false, &mut identities, &mut length), OK);
        let factors = arrays(identities, length);
        assert_eq!(shape(factors[0]).unwrap(), [4, 4]);
        assert_eq!(shape(factors[1]).unwrap(), [4, 0]);
        assert_eq!(values(factors[0])[0], c64::new(1.0, 0.0));

        let empty_wide = array(&[], &[0, 4], 3);
        identities = std::ptr::null_mut();
        assert_eq!(qr(empty_wide, false, &mut identities, &mut length), OK);
        let wide_factors = arrays(identities, length);
        assert_eq!(shape(wide_factors[0]).unwrap(), [0, 0]);
        assert_eq!(shape(wide_factors[1]).unwrap(), [0, 4]);

        let batched = array(
            &[
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0, 0.0),
                c64::new(2.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(2.0, 0.0),
            ],
            &[2, 2, 2],
            3,
        );
        identities = std::ptr::null_mut();
        assert_eq!(qr(batched, true, &mut identities, &mut length), OK);
        let batch_factors = arrays(identities, length);
        assert_eq!(shape(batch_factors[0]).unwrap(), [2, 2, 2]);
        assert_eq!(shape(batch_factors[1]).unwrap(), [2, 2, 2]);
        dispose(&[empty_tall, empty_wide, batched]);
        dispose(&factors);
        dispose(&wide_factors);
        dispose(&batch_factors);
    }
}

#[test]
fn svd_preserves_numpy_dtypes_reconstructs_complex_batches_and_handles_empty_identity() {
    unsafe {
        for (dtype_code, real_dtype) in [(11, 11), (3, 3), (12, 11), (13, 3)] {
            let source_values = if matches!(dtype_code, 12 | 13) {
                [
                    c64::new(1.0, 1.0),
                    c64::new(2.0, -1.0),
                    c64::new(3.0, 0.5),
                    c64::new(4.0, 2.0),
                    c64::new(5.0, -2.0),
                    c64::new(6.0, 1.0),
                ]
            } else {
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(|value| c64::new(value, 0.0))
            };
            let source = array(&source_values, &[3, 2], dtype_code);
            let mut identities = std::ptr::null_mut();
            let mut length = 0;
            assert_eq!(svd(source, false, &mut identities, &mut length), OK);
            let factors = arrays(identities, length);
            assert_eq!(dtype(factors[0]).unwrap(), dtype_code);
            assert_eq!(dtype(factors[1]).unwrap(), real_dtype);
            assert_eq!(dtype(factors[2]).unwrap(), dtype_code);
            let mut scaled_u = values(factors[0]);
            let singular = values(factors[1]);
            for row in 0..3 {
                for column in 0..2 {
                    scaled_u[row * 2 + column] *= singular[column].re;
                }
            }
            let reconstructed = multiply(&scaled_u, 3, 2, &values(factors[2]), 2);
            for (actual, expected) in reconstructed.iter().zip(values(source)) {
                close(*actual, expected);
            }
            dispose(&[source]);
            dispose(&factors);
        }

        let tall = array(&[], &[4, 0], 3);
        let mut identities = std::ptr::null_mut();
        let mut length = 0;
        assert_eq!(svd(tall, true, &mut identities, &mut length), OK);
        let tall_factors = arrays(identities, length);
        assert_eq!(shape(tall_factors[0]).unwrap(), [4, 4]);
        assert_eq!(shape(tall_factors[1]).unwrap(), [0]);
        assert_eq!(shape(tall_factors[2]).unwrap(), [0, 0]);
        assert_eq!(values(tall_factors[0])[15], c64::new(1.0, 0.0));

        let wide = array(&[], &[0, 4], 12);
        identities = std::ptr::null_mut();
        assert_eq!(svd(wide, true, &mut identities, &mut length), OK);
        let wide_factors = arrays(identities, length);
        assert_eq!(shape(wide_factors[0]).unwrap(), [0, 0]);
        assert_eq!(shape(wide_factors[1]).unwrap(), [0]);
        assert_eq!(dtype(wide_factors[1]).unwrap(), 11);
        assert_eq!(shape(wide_factors[2]).unwrap(), [4, 4]);
        assert_eq!(values(wide_factors[2])[15], c64::new(1.0, 0.0));

        let empty_batches = array(&[], &[0, 2, 2], 3);
        identities = std::ptr::null_mut();
        assert_eq!(svd(empty_batches, false, &mut identities, &mut length), OK);
        let batch_factors = arrays(identities, length);
        assert_eq!(shape(batch_factors[0]).unwrap(), [0, 2, 2]);
        assert_eq!(shape(batch_factors[1]).unwrap(), [0, 2]);
        assert_eq!(shape(batch_factors[2]).unwrap(), [0, 2, 2]);
        dispose(&[tall, wide, empty_batches]);
        dispose(&tall_factors);
        dispose(&wide_factors);
        dispose(&batch_factors);
    }
}

#[test]
fn singular_values_cover_no_uv_large_and_full_nullspace_regressions() {
    unsafe {
        for (rows, columns) in [(3usize, 4usize), (4, 4), (4, 3)] {
            for dtype_code in [3, 13] {
                let input_values = vec![
                    if dtype_code == 13 {
                        c64::new(1.0, 1.0)
                    } else {
                        c64::new(1.0, 0.0)
                    };
                    rows * columns
                ];
                let source = array(&input_values, &[rows as i64, columns as i64], dtype_code);
                let mut output = std::ptr::null_mut();
                assert_eq!(singular_values(source, &mut output), OK);
                assert_eq!(shape(output).unwrap(), [rows.min(columns) as i64]);
                assert_eq!(dtype(output).unwrap(), 3);
                assert_eq!(
                    values(output).iter().filter(|value| value.re > 0.5).count(),
                    1,
                );
                dispose(&[source, output]);
            }
        }

        let mut large_values = vec![c64::new(0.0, 0.0); 1000 * 66];
        for index in 0..66 {
            large_values[index * 66 + index] = c64::new(1.0, 0.0);
        }
        let large = array(&large_values, &[1000, 66], 11);
        let mut large_singular = std::ptr::null_mut();
        assert_eq!(singular_values(large, &mut large_singular), OK);
        assert_eq!(shape(large_singular).unwrap(), [66]);
        assert!(values(large_singular)
            .iter()
            .all(|value| (value.re - 1.0).abs() < 1.0e-5));

        let source_values =
            [0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0].map(|value| c64::new(value, 0.0));
        let source = array(&source_values, &[4, 2], 3);
        let mut identities = std::ptr::null_mut();
        let mut length = 0usize;
        assert_eq!(svd(source, true, &mut identities, &mut length), OK);
        let factors = arrays(identities, length);
        assert_eq!(shape(factors[0]).unwrap(), [4, 4]);
        let u = values(factors[0]);
        for null_column in 2..4 {
            for source_column in 0..2 {
                let projection = (0..4)
                    .map(|row| {
                        u[row * 4 + null_column].conj() * source_values[row * 2 + source_column]
                    })
                    .sum::<c64>();
                close(projection, c64::new(0.0, 0.0));
            }
        }
        dispose(&[large, large_singular, source]);
        dispose(&factors);
    }
}

#[test]
fn unsupported_float16_linalg_is_rejected_without_output() {
    unsafe {
        let source = array(
            &[
                c64::new(2.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(2.0, 0.0),
            ],
            &[2, 2],
            14,
        );
        let mut output = std::ptr::null_mut();
        assert_eq!(
            cholesky(source, &mut output),
            crate::package_extension::native_support::UNSUPPORTED_DTYPE,
        );
        assert!(output.is_null());
        dispose(&[source]);
    }
}

#[test]
fn hermitian_eigen_obeys_uplo_batches_empty_shapes_and_numpy_dtype_rules() {
    unsafe {
        for (dtype_code, real_dtype) in [(11, 11), (3, 3), (12, 11), (13, 3)] {
            let complex = matches!(dtype_code, 12 | 13);
            let lower = array(
                &[
                    c64::new(0.0, 0.0),
                    if complex {
                        c64::new(99.0, 99.0)
                    } else {
                        c64::new(99.0, 0.0)
                    },
                    c64::new(1.0, 0.0),
                    c64::new(0.0, 0.0),
                ],
                &[2, 2],
                dtype_code,
            );
            let mut identities = std::ptr::null_mut();
            let mut length = 0;
            assert_eq!(
                symmetric_eigen(lower, false, 0.0, &mut identities, &mut length),
                OK
            );
            let factors = arrays(identities, length);
            assert_eq!(dtype(factors[0]).unwrap(), real_dtype);
            assert_eq!(dtype(factors[1]).unwrap(), dtype_code);
            close(values(factors[0])[0], c64::new(-1.0, 0.0));
            close(values(factors[0])[1], c64::new(1.0, 0.0));

            let upper = array(
                &[
                    c64::new(0.0, 0.0),
                    c64::new(1.0, 0.0),
                    if complex {
                        c64::new(77.0, 77.0)
                    } else {
                        c64::new(77.0, 0.0)
                    },
                    c64::new(0.0, 0.0),
                ],
                &[2, 2],
                dtype_code,
            );
            let mut values_only = std::ptr::null_mut();
            assert_eq!(eigvalsh(upper, true, &mut values_only), OK);
            close(values(values_only)[0], c64::new(-1.0, 0.0));
            close(values(values_only)[1], c64::new(1.0, 0.0));
            dispose(&[lower, upper, values_only]);
            dispose(&factors);
        }

        let empty_batches = array(&[], &[0, 1, 1], 2);
        let mut identities = std::ptr::null_mut();
        let mut length = 0;
        assert_eq!(
            symmetric_eigen(empty_batches, false, 0.0, &mut identities, &mut length),
            OK
        );
        let empty_batch_factors = arrays(identities, length);
        assert_eq!(shape(empty_batch_factors[0]).unwrap(), [0, 1]);
        assert_eq!(shape(empty_batch_factors[1]).unwrap(), [0, 1, 1]);
        assert_eq!(dtype(empty_batch_factors[0]).unwrap(), 3);

        let empty_matrix = array(&[], &[0, 0], 12);
        let mut empty_values = std::ptr::null_mut();
        assert_eq!(eigvalsh(empty_matrix, false, &mut empty_values), OK);
        assert_eq!(shape(empty_values).unwrap(), [0]);
        assert_eq!(dtype(empty_values).unwrap(), 11);

        let nonsquare = array(&[c64::new(1.0, 0.0); 6], &[2, 3], 3);
        let mut rejected = std::ptr::null_mut();
        assert_eq!(eigvalsh(nonsquare, false, &mut rejected), SHAPE_MISMATCH);
        assert!(rejected.is_null());
        assert_eq!(
            symmetric_eigen(
                nonsquare,
                false,
                f64::NAN,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ),
            DOMAIN_ERROR
        );
        dispose(&[empty_batches, empty_matrix, empty_values, nonsquare]);
        dispose(&empty_batch_factors);
    }
}
