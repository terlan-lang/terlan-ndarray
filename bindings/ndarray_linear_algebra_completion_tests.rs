use super::*;

/// Creates one owned Float64 fixture.
unsafe fn array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { owned_real(values, dimensions, 3).unwrap() }
}

/// Creates one owned numerical fixture from canonical complex values.
unsafe fn typed_array(values: &[c64], dimensions: &[i64], dtype: i32) -> Array {
    unsafe { owned_numeric_as(values, dimensions, dtype).unwrap() }
}

/// Reads one result as complex values.
unsafe fn values(array: Array) -> Vec<c64> {
    unsafe { numeric(array, false).unwrap().values }
}

/// Converts and frees one transferred handle array.
unsafe fn arrays(identities: *mut u64, length: usize) -> Vec<Array> {
    let output = unsafe { std::slice::from_raw_parts(identities, length) }
        .iter()
        .map(|&identity| identity as usize as Array)
        .collect();
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(identities) };
    output
}

/// Asserts one real value within numerical tolerance.
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-8,
        "{actual} != {expected}"
    );
}

/// Executes one generalized norm request and returns its owned output.
unsafe fn general_norm(
    array: Array,
    order_kind: i64,
    order_value: f64,
    axes: &[i64],
    keepdims: bool,
) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            terlan_ndarray_v1_norm_general(
                array,
                order_kind,
                order_value,
                axes.as_ptr(),
                axes.len() as i64,
                keepdims,
                &mut output,
            )
        },
        OK
    );
    output
}

#[test]
fn linalg_completion_matches_reference_decompositions_and_tensor_shapes() {
    unsafe {
        let diagonal = array(&[2.0, 0.0, 0.0, 0.5], &[2, 2]);
        let mut scalar = 0.0;
        assert_eq!(terlan_ndarray_v1_cond(diagonal, 2, &mut scalar), OK);
        close(scalar, 4.0);
        assert_eq!(terlan_ndarray_v1_norm(diagonal, &mut scalar), OK);
        close(scalar, 4.25f64.sqrt());

        let mut powered = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_matrix_power(diagonal, -2, &mut powered),
            OK
        );
        let powered_values = values(powered);
        close(powered_values[0].re, 0.25);
        close(powered_values[3].re, 4.0);

        let singular = array(&[1.0, 2.0, 2.0, 4.0], &[2, 2]);
        let mut pseudoinverse = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_pinv(singular, 0.0, &mut pseudoinverse),
            OK
        );
        for (actual, expected) in values(pseudoinverse).iter().zip([0.04, 0.08, 0.08, 0.16]) {
            close(actual.re, expected);
        }

        let rotation = array(&[0.0, -1.0, 1.0, 0.0], &[2, 2]);
        let mut identities = std::ptr::null_mut();
        let mut length = 0usize;
        assert_eq!(
            terlan_ndarray_v1_eig(rotation, &mut identities, &mut length),
            OK
        );
        assert_eq!(length, 2);
        let eigen = arrays(identities, length);
        let mut eigenvalues = values(eigen[0]);
        eigenvalues.sort_by(|lhs, rhs| lhs.im.total_cmp(&rhs.im));
        close(eigenvalues[0].im, -1.0);
        close(eigenvalues[1].im, 1.0);
        let mut values_only = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_eigvals(rotation, &mut values_only), OK);
        assert_eq!(values(values_only).len(), 2);

        let symmetric = array(&[2.0, 1.0, 1.0, 2.0], &[2, 2]);
        let mut symmetric_values = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_eigvalsh(symmetric, &mut symmetric_values),
            OK
        );
        let symmetric_values_read = values(symmetric_values);
        close(symmetric_values_read[0].re, 1.0);
        close(symmetric_values_read[1].re, 3.0);

        identities = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_slogdet(diagonal, &mut identities, &mut length),
            OK
        );
        let signed = arrays(identities, length);
        close(values(signed[0])[0].re, 1.0);
        close(values(signed[1])[0].re, 0.0);

        let left = array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let right = array(&[2.0, 0.0, 0.0, 2.0], &[2, 2]);
        let tail = array(&[1.0, 0.0], &[2]);
        let chain = [
            left as usize as u64,
            right as usize as u64,
            tail as usize as u64,
        ];
        let mut product = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_multi_dot(chain.as_ptr(), 3, &mut product),
            OK
        );
        let product_values = values(product);
        close(product_values[0].re, 2.0);
        close(product_values[1].re, 6.0);

        let tensor_identity = array(
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            &[2, 2, 2, 2],
        );
        let mut tensor_inverse = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_tensorinv(tensor_identity, 2, &mut tensor_inverse),
            OK
        );
        assert_eq!(shape(tensor_inverse).unwrap(), [2, 2, 2, 2]);
        assert_eq!(values(tensor_inverse), values(tensor_identity));

        let rhs = array(&[5.0, 11.0], &[2]);
        let mut solution = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_tensorsolve(left, rhs, std::ptr::null(), 0, &mut solution),
            OK
        );
        let solved = values(solution);
        close(solved[0].re, 1.0);
        close(solved[1].re, 2.0);

        delete_arrays(&[
            diagonal,
            powered,
            singular,
            pseudoinverse,
            rotation,
            values_only,
            symmetric,
            symmetric_values,
            left,
            right,
            tail,
            product,
            tensor_identity,
            tensor_inverse,
            rhs,
            solution,
        ]);
        delete_arrays(&eigen);
        delete_arrays(&signed);
    }
}

#[test]
fn numpy_condition_orders_and_singular_inputs_match() {
    unsafe {
        let matrix = array(&[1.0, 0.0, 1.0, 0.0, -2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3]);
        for (order, expected) in [
            (3, 4.0),
            (-3, 2.0 / 3.0),
            (1, 4.0),
            (-1, 0.5),
            (4, (265.0f64 / 12.0).sqrt()),
        ] {
            let mut actual = 0.0;
            assert_eq!(terlan_ndarray_v1_cond(matrix, order, &mut actual), OK);
            close(actual, expected);
        }

        for singular in [
            array(&[0.0, 0.0, 0.0, 0.0], &[2, 2]),
            array(&[1.0, 1.0, 1.0, 1.0], &[2, 2]),
        ] {
            for order in [0, 1, 2, 4] {
                let mut actual = 0.0;
                assert_eq!(terlan_ndarray_v1_cond(singular, order, &mut actual), OK);
                assert!(actual > 1.0e15);
            }
            for order in [-1, -2] {
                let mut actual = 0.0;
                assert_eq!(terlan_ndarray_v1_cond(singular, order, &mut actual), OK);
            }
            delete_arrays(&[singular]);
        }
        delete_arrays(&[matrix]);
    }
}

#[test]
fn numpy_matrix_power_preserves_dtypes_and_integer_power_rules() {
    unsafe {
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let shift = [
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let shift_squared = [
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ];
        let shift_inverse = [
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let shift_values = shift
            .iter()
            .map(|value| c64::new(*value, 0.0))
            .collect::<Vec<_>>();
        let stacked_values = shift_values
            .iter()
            .chain(shift_values.iter())
            .copied()
            .collect::<Vec<_>>();
        for source_dtype in 1..=14 {
            let source = owned_numeric_as(&shift_values, &[4, 4], source_dtype).unwrap();
            let stacked = owned_numeric_as(&stacked_values, &[2, 4, 4], source_dtype).unwrap();
            for (exponent, expected) in [(0, &identity), (1, &shift), (2, &shift_squared)] {
                let mut powered = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_matrix_power(source, exponent, &mut powered),
                    OK
                );
                assert_eq!(
                    super::super::native_support::dtype(powered).unwrap(),
                    source_dtype
                );
                let actual = values(powered);
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected.iter()) {
                    close(actual.re, *expected);
                    close(actual.im, 0.0);
                }
                delete_arrays(&[powered]);

                let mut stacked_powered = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_matrix_power(stacked, exponent, &mut stacked_powered),
                    OK
                );
                assert_eq!(
                    super::super::native_support::dtype(stacked_powered).unwrap(),
                    source_dtype
                );
                let actual = values(stacked_powered);
                assert_eq!(actual.len(), expected.len() * 2);
                for (index, actual) in actual.iter().enumerate() {
                    close(actual.re, expected[index % expected.len()]);
                    close(actual.im, 0.0);
                }
                delete_arrays(&[stacked_powered]);
            }
            delete_arrays(&[stacked, source]);
        }

        for source_dtype in 1..=13 {
            let source = owned_numeric_as(&shift_values, &[4, 4], source_dtype).unwrap();
            let mut inverse = std::ptr::null_mut();
            assert_eq!(terlan_ndarray_v1_matrix_power(source, -1, &mut inverse), OK);
            let expected_dtype = if matches!(source_dtype, 1 | 2 | 4..=10) {
                3
            } else {
                source_dtype
            };
            assert_eq!(
                super::super::native_support::dtype(inverse).unwrap(),
                expected_dtype
            );
            let actual = values(inverse);
            assert_eq!(actual.len(), shift_inverse.len());
            for (actual, expected) in actual.iter().zip(shift_inverse) {
                close(actual.re, expected);
                close(actual.im, 0.0);
            }
            delete_arrays(&[inverse, source]);
        }
    }
}

#[test]
fn numpy_multi_dot_chains_and_vector_endpoints_match() {
    unsafe {
        let matrix = array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let scale = array(&[2.0, 0.0, 0.0, 2.0], &[2, 2]);
        let column = array(&[1.0, 1.0], &[2, 1]);
        let row = array(&[1.0, 1.0], &[1, 2]);
        let vector = array(&[1.0, 2.0], &[2]);

        for (chain, expected_shape, expected) in [
            (
                vec![matrix as usize as u64, scale as usize as u64],
                vec![2, 2],
                vec![2.0, 4.0, 6.0, 8.0],
            ),
            (
                vec![
                    matrix as usize as u64,
                    scale as usize as u64,
                    column as usize as u64,
                ],
                vec![2, 1],
                vec![6.0, 14.0],
            ),
            (
                vec![
                    matrix as usize as u64,
                    scale as usize as u64,
                    column as usize as u64,
                    row as usize as u64,
                ],
                vec![2, 2],
                vec![6.0, 6.0, 14.0, 14.0],
            ),
            (
                vec![vector as usize as u64, matrix as usize as u64],
                vec![2],
                vec![7.0, 10.0],
            ),
            (
                vec![matrix as usize as u64, vector as usize as u64],
                vec![2],
                vec![5.0, 11.0],
            ),
            (
                vec![
                    vector as usize as u64,
                    matrix as usize as u64,
                    vector as usize as u64,
                ],
                vec![],
                vec![27.0],
            ),
        ] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_multi_dot(chain.as_ptr(), chain.len() as i64, &mut output),
                OK
            );
            assert_eq!(shape(output).unwrap(), expected_shape);
            for (actual, expected) in values(output).iter().zip(expected) {
                close(actual.re, expected);
                close(actual.im, 0.0);
            }
            delete_arrays(&[output]);
        }

        let one = [matrix as usize as u64];
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_multi_dot(one.as_ptr(), 1, &mut output),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());

        for (chain, dimensions, expected) in [
            (
                vec![matrix as usize as u64, scale as usize as u64],
                vec![2, 2],
                vec![2.0, 4.0, 6.0, 8.0],
            ),
            (
                vec![
                    matrix as usize as u64,
                    scale as usize as u64,
                    column as usize as u64,
                ],
                vec![2, 1],
                vec![6.0, 14.0],
            ),
            (
                vec![
                    matrix as usize as u64,
                    scale as usize as u64,
                    column as usize as u64,
                    row as usize as u64,
                ],
                vec![2, 2],
                vec![6.0, 6.0, 14.0, 14.0],
            ),
        ] {
            let destination = array(&vec![0.0; expected.len()], &dimensions);
            assert_eq!(
                terlan_ndarray_v1_multi_dot_into(destination, chain.as_ptr(), chain.len() as i64,),
                OK
            );
            for (actual, expected) in values(destination).iter().zip(expected) {
                close(actual.re, expected);
                close(actual.im, 0.0);
            }
            delete_arrays(&[destination]);
        }

        let wrong_shape = array(&[9.0, 8.0, 7.0], &[3]);
        let chain = [matrix as usize as u64, scale as usize as u64];
        assert_eq!(
            terlan_ndarray_v1_multi_dot_into(wrong_shape, chain.as_ptr(), 2),
            SHAPE_MISMATCH
        );
        assert_eq!(
            values(wrong_shape),
            [9.0, 8.0, 7.0].map(|value| c64::new(value, 0.0))
        );
        delete_arrays(&[wrong_shape, vector, row, column, scale, matrix]);
    }
}

#[test]
fn numpy_general_norm_orders_axes_keepdims_dtypes_and_empty_inputs_match() {
    unsafe {
        let vector = array(&[1.0, 2.0, 3.0, 4.0], &[4]);
        for (kind, order, expected) in [
            (0, 0.0, 30.0f64.sqrt()),
            (1, 0.0, 1.0),
            (2, 0.0, 4.0),
            (5, -2.0, (205.0f64 / 144.0).powf(-0.5)),
            (5, -1.0, 12.0 / 25.0),
            (5, 0.0, 4.0),
            (5, 1.0, 10.0),
            (5, 2.0, 30.0f64.sqrt()),
            (5, 3.0, 100.0f64.powf(1.0 / 3.0)),
            (5, 4.0, 354.0f64.powf(0.25)),
        ] {
            let output = general_norm(vector, kind, order, &[], false);
            close(values(output)[0].re, expected);
            delete_arrays(&[output]);
        }

        let matrix = array(&[1.0, 3.0, 5.0, 7.0], &[2, 2]);
        for (kind, order, expected) in [
            (0, 0.0, 84.0f64.sqrt()),
            (1, 0.0, 4.0),
            (2, 0.0, 12.0),
            (3, 0.0, 84.0f64.sqrt()),
            (4, 0.0, 10.0),
            (5, -2.0, 0.8768943743823404),
            (5, -1.0, 6.0),
            (5, 1.0, 10.0),
            (5, 2.0, 9.123105625617661),
        ] {
            let output = general_norm(matrix, kind, order, &[], false);
            close(values(output)[0].re, expected);
            delete_arrays(&[output]);
        }

        let matrix_3x3 = array(&[0.1, 0.2, 0.3, 0.6, 0.0, 0.5, 0.3, 0.2, 0.1], &[3, 3]);
        for (kind, order, expected) in [
            (0, 0.0, 89.0f64.sqrt() / 10.0),
            (1, 0.0, 0.6),
            (2, 0.0, 1.1),
            (3, 0.0, 89.0f64.sqrt() / 10.0),
            (4, 0.0, 1.3366836911774836),
            (5, -2.0, 0.19456584790481812),
            (5, -1.0, 0.4),
            (5, 1.0, 1.0),
            (5, 2.0, 0.8872294032346128),
        ] {
            let output = general_norm(matrix_3x3, kind, order, &[], false);
            close(values(output)[0].re, expected);
            delete_arrays(&[output]);
        }

        let typed_values = [
            c64::new(1.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(1.0, 0.0),
            c64::new(0.0, 0.0),
            c64::new(1.0, 0.0),
            c64::new(1.0, 0.0),
        ];
        for source_dtype in 1..=14 {
            let typed_vector = owned_numeric_as(&typed_values[..3], &[3], source_dtype).unwrap();
            let typed_matrix = owned_numeric_as(&typed_values, &[2, 3], source_dtype).unwrap();
            let vector_output = general_norm(typed_vector, 5, 1.0, &[], false);
            let matrix_output = general_norm(typed_matrix, 5, 1.0, &[], false);
            let expected_dtype = if matches!(source_dtype, 11 | 12) {
                11
            } else {
                3
            };
            assert_eq!(
                super::super::native_support::dtype(vector_output).unwrap(),
                expected_dtype
            );
            assert_eq!(
                super::super::native_support::dtype(matrix_output).unwrap(),
                expected_dtype
            );
            close(values(vector_output)[0].re, 2.0);
            close(values(matrix_output)[0].re, 2.0);
            delete_arrays(&[matrix_output, vector_output, typed_matrix, typed_vector]);
        }

        let axes = array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let columns = general_norm(axes, 5, 2.0, &[0], false);
        assert_eq!(shape(columns).unwrap(), [3]);
        for (actual, expected) in
            values(columns)
                .iter()
                .zip([17.0f64.sqrt(), 29.0f64.sqrt(), 45.0f64.sqrt()])
        {
            close(actual.re, expected);
        }
        let rows = general_norm(axes, 5, 2.0, &[1], true);
        assert_eq!(shape(rows).unwrap(), [2, 1]);
        for (actual, expected) in values(rows).iter().zip([14.0f64.sqrt(), 77.0f64.sqrt()]) {
            close(actual.re, expected);
        }

        let batched = array(&[1.0, 3.0, 5.0, 7.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);
        let matrix_axes = general_norm(batched, 4, 0.0, &[1, 2], true);
        assert_eq!(shape(matrix_axes).unwrap(), [2, 1, 1]);
        close(values(matrix_axes)[0].re, 10.0);
        close(values(matrix_axes)[1].re, 2.0);

        let complex_values = [c64::new(6.0, 7.0), c64::new(-6.0, 7.0)];
        let complex = owned_numeric_as(&complex_values, &[2], 12).unwrap();
        let complex_norm = general_norm(complex, 5, 3.0, &[], false);
        assert_eq!(
            super::super::native_support::dtype(complex_norm).unwrap(),
            11
        );
        close(values(complex_norm)[0].re, 11.615898132184);

        let minimum = owned_numeric_as(&[c64::new(-2147483648.0, 0.0)], &[1], 6).unwrap();
        let minimum_norm = general_norm(minimum, 5, 3.0, &[], false);
        assert!((values(minimum_norm)[0].re / 2147483648.0 - 1.0).abs() < 1.0e-12);

        let empty = array(&[], &[0]);
        let empty_norm = general_norm(empty, 0, 0.0, &[], true);
        assert_eq!(shape(empty_norm).unwrap(), [1]);
        close(values(empty_norm)[0].re, 0.0);
        let empty_matrix = array(&[], &[1, 0]);
        let empty_matrix_norm = general_norm(empty_matrix, 0, 0.0, &[], false);
        close(values(empty_matrix_norm)[0].re, 0.0);

        delete_arrays(&[
            empty_matrix_norm,
            empty_matrix,
            empty_norm,
            empty,
            minimum_norm,
            minimum,
            complex_norm,
            complex,
            matrix_axes,
            batched,
            rows,
            columns,
            axes,
            matrix_3x3,
            matrix,
            vector,
        ]);
    }
}

#[test]
fn linalg_completion_rejects_bad_orders_shapes_axes_and_singular_inputs() {
    unsafe {
        let vector = array(&[1.0, 2.0], &[2]);
        let singular = array(&[1.0, 2.0, 2.0, 4.0], &[2, 2]);
        let nonsymmetric = array(&[1.0, 2.0, 0.0, 1.0], &[2, 2]);
        let mut scalar = 0.0;
        let mut output = std::ptr::null_mut();
        let mut identities = std::ptr::null_mut();
        let mut length = 0usize;
        let duplicate_axes = [0, 0];
        assert_eq!(
            terlan_ndarray_v1_cond(vector, 2, &mut scalar),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_cond(singular, 9, &mut scalar),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_cond(singular, 2, std::ptr::null_mut()),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_matrix_power(vector, 2, &mut output),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_matrix_power(singular, -1, &mut output),
            DOMAIN_ERROR
        );
        assert_eq!(
            terlan_ndarray_v1_pinv(singular, -1.0, &mut output),
            DOMAIN_ERROR
        );
        assert_eq!(terlan_ndarray_v1_eigvalsh(nonsymmetric, &mut output), OK);
        assert_eq!(shape(output).unwrap(), [2]);
        delete_arrays(&[output]);
        output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_eig(vector, &mut identities, &mut length),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_multi_dot(std::ptr::null(), 0, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_norm_general(vector, 3, 0.0, std::ptr::null(), 0, false, &mut output,),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_norm_general(
                singular,
                5,
                3.0,
                std::ptr::null(),
                0,
                false,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_norm_general(
                singular,
                0,
                0.0,
                std::ptr::null(),
                1,
                false,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        let invalid_axis = [2];
        assert_eq!(
            terlan_ndarray_v1_norm_general(
                vector,
                5,
                2.0,
                invalid_axis.as_ptr(),
                1,
                false,
                &mut output,
            ),
            super::super::native_support::AXIS_OUT_OF_RANGE
        );
        assert_eq!(
            terlan_ndarray_v1_norm_general(
                singular,
                0,
                0.0,
                duplicate_axes.as_ptr(),
                2,
                false,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_norm_general(
                singular,
                9,
                0.0,
                std::ptr::null(),
                0,
                false,
                std::ptr::null_mut(),
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_tensorinv(vector, 0, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_tensorinv(singular, 1, &mut output),
            DOMAIN_ERROR
        );
        assert_eq!(
            terlan_ndarray_v1_tensorsolve(
                singular,
                vector,
                duplicate_axes.as_ptr(),
                2,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert!(identities.is_null());
        delete_arrays(&[vector, singular, nonsymmetric]);
    }
}

#[test]
fn tensor_inverse_matches_numpy_shapes_limits_and_tensor_solve_result() {
    unsafe {
        let mut identity = vec![c64::new(0.0, 0.0); 24 * 24];
        for index in 0..24 {
            identity[index * 24 + index] = c64::new(1.0, 0.0);
        }
        for (dimensions, index, expected_shape) in [
            (&[4, 6, 8, 3][..], 2i64, &[8, 3, 4, 6][..]),
            (&[24, 8, 3][..], 1i64, &[8, 3, 24][..]),
        ] {
            let source = typed_array(&identity, dimensions, 3);
            let mut output = std::ptr::null_mut();
            assert_eq!(terlan_ndarray_v1_tensorinv(source, index, &mut output), OK);
            assert_eq!(shape(output).unwrap(), expected_shape);
            assert_eq!(values(output), identity);
            delete_arrays(&[source, output]);
        }

        let nonsquare_a = array(&vec![1.0; 4 * 6 * 8 * 2], &[4, 6, 8, 2]);
        let nonsquare_b = array(&vec![1.0; 3 * 3 * 2], &[3, 3, 2]);
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_tensorinv(nonsquare_a, 2, &mut rejected),
            SHAPE_MISMATCH,
        );
        assert_eq!(
            terlan_ndarray_v1_tensorinv(nonsquare_b, 1, &mut rejected),
            SHAPE_MISMATCH,
        );

        let source = typed_array(&identity, &[24, 8, 3], 3);
        assert_eq!(
            terlan_ndarray_v1_tensorinv(source, 0, &mut rejected),
            INVALID_ARGUMENT,
        );
        assert_eq!(
            terlan_ndarray_v1_tensorinv(source, -2, &mut rejected),
            INVALID_ARGUMENT,
        );
        let rhs = array(&vec![1.0; 24], &[24]);
        let mut solution = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_tensorsolve(source, rhs, std::ptr::null(), 0, &mut solution),
            OK,
        );
        assert_eq!(shape(solution).unwrap(), [8, 3]);
        assert_eq!(values(solution), vec![c64::new(1.0, 0.0); 24]);
        assert!(rejected.is_null());
        delete_arrays(&[nonsquare_a, nonsquare_b, source, rhs, solution]);
    }
}
