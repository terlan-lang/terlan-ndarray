use super::*;
use crate::package_extension::native_support::{dtype, numeric, owned_numeric_as, shape};

/// Creates one owned array with an explicit numerical dtype.
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

/// Disposes every live test array exactly once.
unsafe fn dispose(arrays: &[Array]) {
    for &array in arrays {
        if !array.is_null() {
            assert_eq!(unsafe { crate::ffi::terlan_ndarray_v1_delete(array) }, OK);
        }
    }
}

/// Returns one row-major identity matrix.
fn identity(dimension: usize) -> Vec<c64> {
    (0..dimension * dimension)
        .map(|index| {
            c64::new(
                if index / dimension == index % dimension {
                    1.0
                } else {
                    0.0
                },
                0.0,
            )
        })
        .collect()
}

/// Builds a tall matrix with one or two exact dependent columns.
fn reduced_rank_values(second_dependency: bool) -> Vec<c64> {
    let mut output = vec![c64::new(0.0, 0.0); 40 * 10];
    for column in [1usize, 2, 3, 4, 5, 6, 7, 8, 9] {
        output[column * 10 + column] = c64::new(1.0, 0.0);
    }
    for row in 0..40 {
        output[row * 10] = output[row * 10 + 1] + output[row * 10 + 2];
        if second_dependency {
            output[row * 10 + 3] = output[row * 10 + 4] + output[row * 10 + 5];
        }
    }
    output
}

#[test]
fn matrix_rank_covers_scalars_vectors_stacks_hermitian_tolerance_and_reduced_rank() {
    unsafe {
        let full = array(&identity(4), &[4, 4], 3);
        let mut deficient_values = identity(4);
        deficient_values[15] = c64::new(0.0, 0.0);
        let deficient = array(&deficient_values, &[4, 4], 3);
        let zero = array(&[c64::new(0.0, 0.0); 16], &[4, 4], 3);
        let stack_values = deficient_values
            .iter()
            .chain(identity(4).iter())
            .chain([c64::new(0.0, 0.0); 16].iter())
            .copied()
            .collect::<Vec<_>>();
        let stack = array(&stack_values, &[3, 4, 4], 3);
        let vector = array(&[c64::new(1.0, 0.0), c64::new(0.0, 0.0)], &[2], 3);
        let scalar = array(&[c64::new(1.0, 0.0)], &[], 3);
        let mut full_rank = std::ptr::null_mut();
        let mut deficient_rank = std::ptr::null_mut();
        let mut zero_rank = std::ptr::null_mut();
        let mut stack_rank = std::ptr::null_mut();
        let mut vector_rank = std::ptr::null_mut();
        let mut scalar_rank = std::ptr::null_mut();
        assert_eq!(matrix_rank(full, 0.0, false, &mut full_rank), OK);
        assert_eq!(matrix_rank(deficient, 0.0, false, &mut deficient_rank), OK);
        assert_eq!(matrix_rank(zero, 0.0, false, &mut zero_rank), OK);
        assert_eq!(matrix_rank(stack, 0.0, false, &mut stack_rank), OK);
        assert_eq!(matrix_rank(vector, 0.0, false, &mut vector_rank), OK);
        assert_eq!(matrix_rank(scalar, 0.0, false, &mut scalar_rank), OK);
        assert_eq!(values(full_rank), [c64::new(4.0, 0.0)]);
        assert_eq!(values(deficient_rank), [c64::new(3.0, 0.0)]);
        assert_eq!(values(zero_rank), [c64::new(0.0, 0.0)]);
        assert_eq!(
            values(stack_rank),
            [c64::new(3.0, 0.0), c64::new(4.0, 0.0), c64::new(0.0, 0.0),]
        );
        assert_eq!(values(vector_rank), [c64::new(1.0, 0.0)]);
        assert_eq!(values(scalar_rank), [c64::new(1.0, 0.0)]);

        let almost = array(
            &[
                c64::new(1.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(0.0, 0.0),
                c64::new(1.0e-8, 0.0),
            ],
            &[2, 2],
            3,
        );
        let mut above = std::ptr::null_mut();
        let mut below = std::ptr::null_mut();
        assert_eq!(matrix_rank(almost, 0.99e-8, true, &mut above), OK);
        assert_eq!(matrix_rank(almost, 1.01e-8, true, &mut below), OK);
        assert_eq!(values(above), [c64::new(2.0, 0.0)]);
        assert_eq!(values(below), [c64::new(1.0, 0.0)]);

        let rank_nine = array(&reduced_rank_values(false), &[40, 10], 3);
        let rank_eight = array(&reduced_rank_values(true), &[40, 10], 3);
        let mut observed_nine = std::ptr::null_mut();
        let mut observed_eight = std::ptr::null_mut();
        assert_eq!(matrix_rank(rank_nine, 0.0, false, &mut observed_nine), OK);
        assert_eq!(matrix_rank(rank_eight, 0.0, false, &mut observed_eight), OK);
        assert_eq!(values(observed_nine), [c64::new(9.0, 0.0)]);
        assert_eq!(values(observed_eight), [c64::new(8.0, 0.0)]);
        dispose(&[
            full,
            deficient,
            zero,
            stack,
            vector,
            scalar,
            full_rank,
            deficient_rank,
            zero_rank,
            stack_rank,
            vector_rank,
            scalar_rank,
            almost,
            above,
            below,
            rank_nine,
            rank_eight,
            observed_nine,
            observed_eight,
        ]);
    }
}

#[test]
fn lstsq_covers_numpy_empty_shapes_residuals_dtypes_and_incompatible_dimensions() {
    unsafe {
        for (rows, columns, rhs_columns) in [
            (4usize, 2usize, 2usize),
            (0, 4, 1),
            (0, 4, 2),
            (4, 0, 1),
            (4, 0, 2),
            (4, 2, 0),
            (0, 0, 0),
        ] {
            let coefficient_values = (0..rows * columns)
                .map(|index| c64::new(index as f64, 0.0))
                .collect::<Vec<_>>();
            let rhs_values = vec![c64::new(1.0, 0.0); rows * rhs_columns];
            let coefficients = array(&coefficient_values, &[rows as i64, columns as i64], 7);
            let rhs = array(&rhs_values, &[rows as i64, rhs_columns as i64], 7);
            let mut identities = std::ptr::null_mut();
            let mut length = 0usize;
            assert_eq!(
                least_squares(coefficients, rhs, 0.0, &mut identities, &mut length),
                OK
            );
            assert_eq!(length, 4);
            let output = arrays(identities, length);
            assert_eq!(
                shape(output[0]).unwrap(),
                [columns as i64, rhs_columns as i64]
            );
            let residual_length = if rows > columns { rhs_columns } else { 0 };
            assert_eq!(shape(output[1]).unwrap(), [residual_length as i64]);
            assert_eq!(shape(output[2]).unwrap(), []);
            assert_eq!(shape(output[3]).unwrap(), [rows.min(columns) as i64]);
            assert_eq!(dtype(output[0]).unwrap(), 3);
            assert_eq!(dtype(output[1]).unwrap(), 3);
            assert_eq!(dtype(output[2]).unwrap(), 7);
            assert_eq!(dtype(output[3]).unwrap(), 3);
            assert_eq!(values(output[2]), [c64::new(rows.min(columns) as f64, 0.0)]);
            if rows == 0 {
                assert!(values(output[0])
                    .iter()
                    .all(|value| *value == c64::new(0.0, 0.0)));
            }
            if rows > columns && columns == 0 {
                assert_eq!(
                    values(output[1]),
                    vec![c64::new(rows as f64, 0.0); rhs_columns]
                );
            }
            dispose(&[
                coefficients,
                rhs,
                output[0],
                output[1],
                output[2],
                output[3],
            ]);
        }

        for dtype_code in [11, 3, 12, 13] {
            let coefficients = array(
                &[
                    c64::new(1.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(0.0, 0.0),
                    c64::new(1.0, 0.0),
                    c64::new(1.0, 0.0),
                    c64::new(1.0, 0.0),
                ],
                &[3, 2],
                dtype_code,
            );
            let rhs = array(
                &[c64::new(1.0, 0.0), c64::new(2.0, 0.0), c64::new(3.0, 0.0)],
                &[3],
                dtype_code,
            );
            let mut identities = std::ptr::null_mut();
            let mut length = 0usize;
            assert_eq!(
                least_squares(coefficients, rhs, 0.0, &mut identities, &mut length),
                OK
            );
            let output = arrays(identities, length);
            assert_eq!(dtype(output[0]).unwrap(), dtype_code);
            assert_eq!(
                dtype(output[1]).unwrap(),
                if matches!(dtype_code, 12 | 13) {
                    if dtype_code == 12 {
                        11
                    } else {
                        3
                    }
                } else {
                    dtype_code
                }
            );
            assert_eq!(values(output[2]), [c64::new(2.0, 0.0)]);
            dispose(&[
                coefficients,
                rhs,
                output[0],
                output[1],
                output[2],
                output[3],
            ]);
        }

        let coefficients = array(&identity(4)[..8], &[4, 2], 3);
        let incompatible = array(&[c64::new(1.0, 0.0); 5], &[5], 3);
        let mut rejected = std::ptr::null_mut();
        let mut rejected_length = 0usize;
        assert_eq!(
            least_squares(
                coefficients,
                incompatible,
                0.0,
                &mut rejected,
                &mut rejected_length,
            ),
            SHAPE_MISMATCH
        );
        assert!(rejected.is_null());
        assert_eq!(rejected_length, 0);
        dispose(&[coefficients, incompatible]);
    }
}

#[test]
fn complex_lstsq_handles_more_right_hand_sides_than_unknowns() {
    unsafe {
        let size = 20usize;
        let rhs_columns = 70usize;
        let mut coefficients = vec![c64::new(0.0, 0.0); size * size];
        for index in 0..size {
            coefficients[index * size + index] = c64::new(index as f64 + 1.0, 0.5);
        }
        let expected = (0..size * rhs_columns)
            .map(|index| c64::new((index % 17) as f64 - 8.0, (index % 11) as f64 - 5.0))
            .collect::<Vec<_>>();
        let mut rhs = vec![c64::new(0.0, 0.0); size * rhs_columns];
        for row in 0..size {
            for column in 0..rhs_columns {
                rhs[row * rhs_columns + column] =
                    coefficients[row * size + row] * expected[row * rhs_columns + column];
            }
        }
        let coefficients_array = array(&coefficients, &[size as i64, size as i64], 13);
        let rhs_array = array(&rhs, &[size as i64, rhs_columns as i64], 13);
        let mut identities = std::ptr::null_mut();
        let mut length = 0usize;
        assert_eq!(
            least_squares(
                coefficients_array,
                rhs_array,
                0.0,
                &mut identities,
                &mut length,
            ),
            OK,
        );
        let output = arrays(identities, length);
        assert_eq!(shape(output[0]).unwrap(), [size as i64, rhs_columns as i64]);
        assert_eq!(dtype(output[0]).unwrap(), 13);
        for (actual, expected) in values(output[0]).iter().zip(expected) {
            assert!((*actual - expected).norm() < 1.0e-5);
        }
        assert_eq!(shape(output[1]).unwrap(), [0]);
        assert_eq!(values(output[2]), [c64::new(size as f64, 0.0)]);
        dispose(&[
            coefficients_array,
            rhs_array,
            output[0],
            output[1],
            output[2],
            output[3],
        ]);
    }
}
