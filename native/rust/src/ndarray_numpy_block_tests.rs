use super::*;

/// Creates one Float64 fixture with a checked shape.
unsafe fn array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, dimensions, 3).unwrap() }
}

/// Executes the native block ABI for borrowed leaves and prefix tokens.
unsafe fn block(leaves: &[Array], tokens: &[i64]) -> Result<Array, i32> {
    let identities = leaves
        .iter()
        .map(|array| *array as usize as u64)
        .collect::<Vec<_>>();
    let mut output = std::ptr::null_mut();
    let status = unsafe {
        terlan_ndarray_v1_block(
            identities.as_ptr(),
            identities.len() as i64,
            tokens.as_ptr(),
            tokens.len() as i64,
            &mut output,
        )
    };
    if status == OK {
        Ok(output)
    } else {
        assert!(output.is_null());
        Err(status)
    }
}

/// Reads one result's logical values.
unsafe fn values(array: Array) -> Vec<f64> {
    unsafe { super::super::native_support::floats(array, false).unwrap() }
}

/// Disposes every supplied array.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { delete_arrays(arrays) };
}

#[test]
fn block_matches_copy_row_column_vector_and_mixed_rank_semantics() {
    unsafe {
        let scalar = array(&[5.0], &[]);
        let vector_a = array(&[1.0, 2.0, 3.0], &[3]);
        let vector_b = array(&[2.0, 3.0, 4.0], &[3]);
        let vector_two = array(&[2.0, 2.0], &[2]);
        let matrix = array(&[1.0; 4], &[2, 2]);
        let doubled = array(&[2.0; 4], &[2, 2]);
        let scalar_copy = block(&[scalar], &[-1]).unwrap();
        let row = block(&[matrix, doubled], &[2, -1, -1]).unwrap();
        let column = block(&[matrix, doubled], &[2, 1, -1, 1, -1]).unwrap();
        let vectors = block(&[vector_a, vector_b], &[2, -1, -1]).unwrap();
        let vector_rows = block(
            &[vector_a, vector_b, vector_a, vector_b],
            &[2, 2, -1, -1, 2, -1, -1],
        )
        .unwrap();
        let vector_columns = block(&[vector_a, vector_b], &[2, 1, -1, 1, -1]).unwrap();
        let mixed = block(&[matrix, vector_two], &[2, 1, -1, 1, -1]).unwrap();

        assert_eq!(shape(scalar_copy).unwrap(), []);
        assert_eq!(shape(row).unwrap(), [2, 4]);
        assert_eq!(values(row), [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]);
        assert_eq!(shape(column).unwrap(), [4, 2]);
        assert_eq!(shape(vectors).unwrap(), [6]);
        assert_eq!(values(vectors), [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape(vector_rows).unwrap(), [2, 6]);
        assert_eq!(shape(vector_columns).unwrap(), [2, 3]);
        assert_eq!(shape(mixed).unwrap(), [3, 2]);
        assert_eq!(values(mixed), [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]);

        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_float(scalar_copy, std::ptr::null(), 0, 9.0),
            OK
        );
        assert_eq!(values(scalar), [5.0]);
        dispose(&[
            mixed,
            vector_columns,
            vector_rows,
            vectors,
            column,
            row,
            scalar_copy,
            doubled,
            matrix,
            vector_b,
            vector_a,
            vector_two,
            scalar,
        ]);
    }
}

#[test]
fn block_matches_complicated_nested_and_three_dimensional_numpy_layouts() {
    unsafe {
        let one = array(&[1.0, 1.0, 1.0], &[1, 3]);
        let two = array(&[2.0, 2.0, 2.0], &[1, 3]);
        let three = array(&[3.0; 6], &[1, 6]);
        let four = array(&[4.0; 6], &[6]);
        let five = array(&[5.0], &[]);
        let six = array(&[6.0; 5], &[5]);
        let zero = array(&[0.0; 12], &[2, 6]);
        let complicated = block(
            &[one, two, three, four, five, six, zero],
            &[5, 2, -1, -1, 1, -1, 1, -1, 2, -1, -1, 1, -1],
        )
        .unwrap();
        assert_eq!(shape(complicated).unwrap(), [6, 6]);
        assert_eq!(
            values(complicated),
            [
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0,
                4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ]
        );

        let nested_one = array(&[1.0; 3], &[3]);
        let nested_three = array(&[3.0; 3], &[3]);
        let nested_four = array(&[4.0; 3], &[3]);
        let left = block(
            &[nested_one, nested_three, nested_four],
            &[3, 1, -1, 1, -1, 1, -1],
        )
        .unwrap();
        let square = array(&[2.0; 9], &[3, 3]);
        let nested = block(
            &[left, square, five, six, zero],
            &[3, 2, -1, -1, 2, -1, -1, 1, -1],
        )
        .unwrap();
        assert_eq!(shape(nested).unwrap(), [6, 6]);

        let mut cubes = Vec::new();
        for (value, dimensions) in [
            (1.0, [2, 2, 2]),
            (4.0, [2, 2, 3]),
            (3.0, [2, 3, 2]),
            (5.0, [2, 3, 3]),
            (2.0, [3, 2, 2]),
            (6.0, [3, 2, 3]),
            (7.0, [3, 3, 2]),
            (8.0, [3, 3, 3]),
        ] {
            let count = dimensions.iter().product::<i64>() as usize;
            cubes.push(array(&vec![value; count], &dimensions));
        }
        let cube = block(
            &cubes,
            &[2, 2, 2, -1, -1, 2, -1, -1, 2, 2, -1, -1, 2, -1, -1],
        )
        .unwrap();
        assert_eq!(shape(cube).unwrap(), [5, 5, 5]);
        assert_eq!(
            &values(cube)[..10],
            [1.0, 1.0, 4.0, 4.0, 4.0, 1.0, 1.0, 4.0, 4.0, 4.0]
        );

        dispose(&[cube, nested, square, left, complicated]);
        dispose(&cubes);
        dispose(&[
            nested_four,
            nested_three,
            nested_one,
            zero,
            six,
            five,
            four,
            three,
            two,
            one,
        ]);
    }
}

#[test]
fn block_promotes_rank_and_dtype_and_rejects_shape_depth_and_encoding_errors() {
    unsafe {
        let scalar = array(&[1.0], &[]);
        let matrix = array(&[2.0, 2.0], &[1, 2]);
        let cube = array(&[3.0, 3.0, 3.0], &[1, 1, 3]);
        let different = block(&[scalar, matrix, cube], &[3, -1, -1, -1]).unwrap();
        assert_eq!(shape(different).unwrap(), [1, 1, 6]);
        assert_eq!(values(different), [1.0, 2.0, 2.0, 3.0, 3.0, 3.0]);

        let deep_matrix = array(&[3.0; 6], &[1, 2, 3]);
        let depths = block(&[scalar, matrix, deep_matrix], &[2, 2, -1, -1, 1, -1]).unwrap();
        assert_eq!(shape(depths).unwrap(), [1, 3, 3]);

        let mut narrow = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_from_floats_as(
                [4.0_f64].as_ptr(),
                1,
                [1_i64].as_ptr(),
                1,
                11,
                &mut narrow,
            ),
            OK
        );
        let promoted = block(&[narrow, matrix], &[2, -1, -1]).unwrap();
        assert_eq!(super::super::native_support::dtype(promoted).unwrap(), 3);

        let short = array(&[0.0, 0.0], &[2]);
        let square = array(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        assert_eq!(block(&[short, square], &[2, -1, -1]), Err(SHAPE_MISMATCH));
        assert_eq!(block(&[square, short], &[2, -1, -1]), Err(SHAPE_MISMATCH));
        assert_eq!(
            block(&[short, square], &[2, -1, 1, -1]),
            Err(SHAPE_MISMATCH)
        );
        assert_eq!(block(&[], &[0]), Err(SHAPE_MISMATCH));
        assert_eq!(block(&[short], &[2, -1]), Err(INVALID_ARGUMENT));
        assert_eq!(block(&[short], &[-1, -1]), Err(INVALID_ARGUMENT));
        assert_eq!(block(&[short, square], &[-1]), Err(INVALID_ARGUMENT));

        let mut output = square;
        assert_eq!(
            terlan_ndarray_v1_block(std::ptr::null(), 0, std::ptr::null(), 0, &mut output),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert_eq!(
            terlan_ndarray_v1_block(
                std::ptr::null(),
                0,
                [-1_i64].as_ptr(),
                1,
                std::ptr::null_mut()
            ),
            INVALID_ARGUMENT
        );
        dispose(&[
            square,
            short,
            promoted,
            narrow,
            depths,
            deep_matrix,
            different,
            cube,
            matrix,
            scalar,
        ]);
    }
}

#[test]
fn block_preserves_fortran_order_only_when_every_leaf_is_fortran_only() {
    unsafe {
        let source_a = array(&[1.0; 6], &[2, 3]);
        let source_b = array(&[2.0; 6], &[2, 3]);
        let mut fortran_a = std::ptr::null_mut();
        let mut fortran_b = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(source_a, 1, &mut fortran_a),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(source_b, 1, &mut fortran_b),
            OK
        );
        let fortran = block(
            &[fortran_a, fortran_b, fortran_a, fortran_b],
            &[2, 2, -1, -1, 2, -1, -1],
        )
        .unwrap();
        let mixed = block(
            &[fortran_a, source_b, fortran_a, source_b],
            &[2, 2, -1, -1, 2, -1, -1],
        )
        .unwrap();
        let mut c_contiguous = 0;
        let mut f_contiguous = 0;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(fortran, &mut f_contiguous),
            OK
        );
        assert_eq!(f_contiguous, 1);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_contiguous(mixed, &mut c_contiguous),
            OK
        );
        assert_eq!(c_contiguous, 1);
        dispose(&[mixed, fortran, fortran_b, fortran_a, source_b, source_a]);
    }
}
