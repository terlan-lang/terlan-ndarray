use super::super::numpy_shape_extension::*;
use super::*;

const DTYPE_MISMATCH: i32 = 106;

unsafe fn array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, dimensions, 3).unwrap() }
}

unsafe fn values(array: Array) -> Vec<f64> {
    unsafe { super::super::native_support::floats(array, false).unwrap() }
}

unsafe fn take_arrays(raw: *mut u64, length: usize) -> Vec<Array> {
    let arrays = if length == 0 {
        Vec::new()
    } else {
        assert!(!raw.is_null());
        unsafe { std::slice::from_raw_parts(raw, length) }
            .iter()
            .map(|handle| *handle as usize as Array)
            .collect()
    };
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(raw) };
    arrays
}

#[test]
fn atleast_dimensions_match_numpy_scalar_vector_matrix_and_cube() {
    unsafe {
        let scalar = array(&[7.0], &[]);
        let vector = array(&[1.0, 2.0], &[2]);
        let matrix = array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let cube = array(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let inputs = [scalar, vector, matrix, cube];
        let expected = [
            [vec![1], vec![1, 1], vec![1, 1, 1]],
            [vec![2], vec![1, 2], vec![1, 2, 1]],
            [vec![2, 2], vec![2, 2], vec![2, 2, 1]],
            [vec![1, 2, 2], vec![1, 2, 2], vec![1, 2, 2]],
        ];
        let mut outputs = Vec::new();
        for (input_index, input) in inputs.into_iter().enumerate() {
            for rank in 1..=3 {
                let mut output = std::ptr::null_mut();
                let status = match rank {
                    1 => crate::ffi::terlan_ndarray_v1_atleast_1d(input, &mut output),
                    2 => crate::ffi::terlan_ndarray_v1_atleast_2d(input, &mut output),
                    3 => crate::ffi::terlan_ndarray_v1_atleast_3d(input, &mut output),
                    _ => unreachable!(),
                };
                assert_eq!(status, OK);
                assert_eq!(shape(output).unwrap(), expected[input_index][rank - 1]);
                let mut shares = false;
                assert_eq!(
                    crate::ffi::terlan_ndarray_v1_shares_memory(input, output, &mut shares),
                    OK
                );
                assert!(shares);
                outputs.push(output);
            }
        }
        super::super::native_support::delete_arrays(&outputs);
        super::super::native_support::delete_arrays(&[scalar, vector, matrix, cube]);
    }
}

#[test]
fn alen_matches_numpy_for_scalars_vectors_matrices_and_empty_axes() {
    unsafe {
        let scalar = array(&[7.0], &[]);
        let empty = array(&[], &[0, 3]);
        let matrix = array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mut length = -1;

        assert_eq!(terlan_ndarray_v1_alen(matrix, &mut length), OK);
        assert_eq!(length, 2);
        assert_eq!(terlan_ndarray_v1_alen(empty, &mut length), OK);
        assert_eq!(length, 0);
        assert_eq!(terlan_ndarray_v1_alen(scalar, &mut length), OK);
        assert_eq!(length, 1);
        assert_eq!(
            terlan_ndarray_v1_alen(matrix, std::ptr::null_mut()),
            INVALID_ARGUMENT
        );

        super::super::native_support::delete_arrays(&[matrix, empty, scalar]);
    }
}

#[test]
fn stacking_and_designated_splits_match_numpy_rank_rules() {
    unsafe {
        let scalar_a = array(&[1.0], &[]);
        let scalar_b = array(&[2.0], &[]);
        let scalars = [scalar_a as usize as u64, scalar_b as usize as u64];
        let vector_a = array(&[1.0, 2.0], &[2]);
        let vector_b = array(&[3.0, 4.0], &[2]);
        let vectors = [vector_a as usize as u64, vector_b as usize as u64];
        let matrix_a = array(&[1.0, 2.0], &[2, 1]);
        let matrix_b = array(&[3.0, 4.0], &[2, 1]);
        let matrices = [matrix_a as usize as u64, matrix_b as usize as u64];
        let mut scalar_horizontal = std::ptr::null_mut();
        let mut scalar_vertical = std::ptr::null_mut();
        let mut scalar_depth = std::ptr::null_mut();
        let mut horizontal = std::ptr::null_mut();
        let mut vertical = std::ptr::null_mut();
        let mut depth = std::ptr::null_mut();
        let mut columns = std::ptr::null_mut();
        let mut matrix_horizontal = std::ptr::null_mut();
        let mut matrix_vertical = std::ptr::null_mut();
        let mut matrix_depth = std::ptr::null_mut();
        let mut matrix_columns = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_hstack(scalars.as_ptr(), 2, &mut scalar_horizontal),
            OK
        );
        assert_eq!(shape(scalar_horizontal).unwrap(), [2]);
        assert_eq!(values(scalar_horizontal), [1.0, 2.0]);
        assert_eq!(
            terlan_ndarray_v1_vstack(scalars.as_ptr(), 2, &mut scalar_vertical),
            OK
        );
        assert_eq!(shape(scalar_vertical).unwrap(), [2, 1]);
        assert_eq!(
            terlan_ndarray_v1_dstack(scalars.as_ptr(), 2, &mut scalar_depth),
            OK
        );
        assert_eq!(shape(scalar_depth).unwrap(), [1, 1, 2]);
        assert_eq!(
            terlan_ndarray_v1_hstack(vectors.as_ptr(), 2, &mut horizontal),
            OK
        );
        assert_eq!(shape(horizontal).unwrap(), [4]);
        assert_eq!(
            terlan_ndarray_v1_vstack(vectors.as_ptr(), 2, &mut vertical),
            OK
        );
        assert_eq!(shape(vertical).unwrap(), [2, 2]);
        assert_eq!(
            terlan_ndarray_v1_dstack(vectors.as_ptr(), 2, &mut depth),
            OK
        );
        assert_eq!(shape(depth).unwrap(), [1, 2, 2]);
        assert_eq!(
            terlan_ndarray_v1_column_stack(vectors.as_ptr(), 2, &mut columns),
            OK
        );
        assert_eq!(shape(columns).unwrap(), [2, 2]);
        assert_eq!(values(columns), [1.0, 3.0, 2.0, 4.0]);
        assert_eq!(
            terlan_ndarray_v1_hstack(matrices.as_ptr(), 2, &mut matrix_horizontal),
            OK
        );
        assert_eq!(shape(matrix_horizontal).unwrap(), [2, 2]);
        assert_eq!(values(matrix_horizontal), [1.0, 3.0, 2.0, 4.0]);
        assert_eq!(
            terlan_ndarray_v1_vstack(matrices.as_ptr(), 2, &mut matrix_vertical),
            OK
        );
        assert_eq!(shape(matrix_vertical).unwrap(), [4, 1]);
        assert_eq!(
            terlan_ndarray_v1_dstack(matrices.as_ptr(), 2, &mut matrix_depth),
            OK
        );
        assert_eq!(shape(matrix_depth).unwrap(), [2, 1, 2]);
        assert_eq!(
            terlan_ndarray_v1_column_stack(matrices.as_ptr(), 2, &mut matrix_columns),
            OK
        );
        assert_eq!(shape(matrix_columns).unwrap(), [2, 2]);

        let split_vector = array(&[0.0, 1.0, 2.0, 3.0], &[4]);
        let split_matrix = array(&(0..8).map(f64::from).collect::<Vec<_>>(), &[2, 4]);
        let split_cube = array(&(0..16).map(f64::from).collect::<Vec<_>>(), &[2, 2, 4]);
        let mut raw = std::ptr::null_mut();
        let mut length = 0usize;
        assert_eq!(
            terlan_ndarray_v1_hsplit(split_vector, 2, &mut raw, &mut length),
            OK
        );
        let horizontal_vector_parts = take_arrays(raw, length);
        assert_eq!(horizontal_vector_parts.len(), 2);
        assert_eq!(shape(horizontal_vector_parts[0]).unwrap(), [2]);
        assert_eq!(values(horizontal_vector_parts[1]), [2.0, 3.0]);

        raw = std::ptr::null_mut();
        length = 0;
        assert_eq!(
            terlan_ndarray_v1_hsplit(split_matrix, 2, &mut raw, &mut length),
            OK
        );
        let horizontal_matrix_parts = take_arrays(raw, length);
        assert_eq!(shape(horizontal_matrix_parts[0]).unwrap(), [2, 2]);
        assert_eq!(values(horizontal_matrix_parts[1]), [2.0, 3.0, 6.0, 7.0]);

        raw = std::ptr::null_mut();
        length = 0;
        assert_eq!(
            terlan_ndarray_v1_vsplit(split_matrix, 2, &mut raw, &mut length),
            OK
        );
        let vertical_parts = take_arrays(raw, length);
        assert_eq!(shape(vertical_parts[0]).unwrap(), [1, 4]);

        raw = std::ptr::null_mut();
        length = 0;
        assert_eq!(
            terlan_ndarray_v1_dsplit(split_cube, 2, &mut raw, &mut length),
            OK
        );
        let depth_parts = take_arrays(raw, length);
        assert_eq!(shape(depth_parts[0]).unwrap(), [2, 2, 2]);

        let mut rejected = std::ptr::null_mut();
        let mut rejected_length = 0usize;
        assert_eq!(
            terlan_ndarray_v1_hsplit(scalar_a, 1, &mut rejected, &mut rejected_length),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_vsplit(vector_a, 1, &mut rejected, &mut rejected_length),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_dsplit(split_matrix, 1, &mut rejected, &mut rejected_length),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_vsplit(scalar_a, 1, &mut rejected, &mut rejected_length),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_dsplit(scalar_a, 1, &mut rejected, &mut rejected_length),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_dsplit(vector_a, 1, &mut rejected, &mut rejected_length),
            SHAPE_MISMATCH
        );
        assert!(rejected.is_null());

        let mut owned = vec![
            scalar_a,
            scalar_b,
            vector_a,
            vector_b,
            matrix_a,
            matrix_b,
            scalar_horizontal,
            scalar_vertical,
            scalar_depth,
            horizontal,
            vertical,
            depth,
            columns,
            matrix_horizontal,
            matrix_vertical,
            matrix_depth,
            matrix_columns,
            split_vector,
            split_matrix,
            split_cube,
        ];
        owned.extend(horizontal_vector_parts);
        owned.extend(horizontal_matrix_parts);
        owned.extend(vertical_parts);
        owned.extend(depth_parts);
        super::super::native_support::delete_arrays(&owned);
    }
}

#[test]
fn stacking_append_diagonal_and_index_shapes_match_numpy_contracts() {
    unsafe {
        let lhs = array(&[1.0, 2.0], &[2]);
        let rhs = array(&[3.0, 4.0], &[2]);
        let identities = [lhs as usize as u64, rhs as usize as u64];
        let mut vertical = std::ptr::null_mut();
        let mut columns = std::ptr::null_mut();
        let mut appended = std::ptr::null_mut();
        let mut diagonal = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_vstack(identities.as_ptr(), 2, &mut vertical),
            0
        );
        assert_eq!(shape(vertical).unwrap(), [2, 2]);
        assert_eq!(
            terlan_ndarray_v1_column_stack(identities.as_ptr(), 2, &mut columns),
            0
        );
        assert_eq!(shape(columns).unwrap(), [2, 2]);
        assert_eq!(terlan_ndarray_v1_append(lhs, rhs, -1, &mut appended), 0);
        assert_eq!(values(appended), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(terlan_ndarray_v1_diag(lhs, 1, &mut diagonal), 0);
        assert_eq!(shape(diagonal).unwrap(), [3, 3]);
        super::super::native_support::delete_arrays(&[
            lhs, rhs, vertical, columns, appended, diagonal,
        ]);
    }
}

#[test]
fn diag_preserves_dtype_offsets_bounds_fortran_views_and_rank_rejections() {
    unsafe {
        let vector =
            super::super::native_support::owned_int_as(&[i64::MIN, 0, i64::MAX], &[3], 2).unwrap();
        let mut centered = std::ptr::null_mut();
        let mut upper = std::ptr::null_mut();
        let mut lower = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_diag(vector, 0, &mut centered), OK);
        assert_eq!(shape(centered).unwrap(), [3, 3]);
        assert_eq!(super::super::native_support::dtype(centered).unwrap(), 2);
        assert_eq!(
            super::super::native_support::ints(centered).unwrap(),
            [i64::MIN, 0, 0, 0, 0, 0, 0, 0, i64::MAX]
        );
        assert_eq!(terlan_ndarray_v1_diag(vector, 2, &mut upper), OK);
        assert_eq!(shape(upper).unwrap(), [5, 5]);
        assert_eq!(terlan_ndarray_v1_diag(vector, -2, &mut lower), OK);
        assert_eq!(shape(lower).unwrap(), [5, 5]);

        let matrix =
            super::super::native_support::owned_int_as(&[1, 2, 3, 4, 5, 6], &[3, 2], 2).unwrap();
        let mut main = std::ptr::null_mut();
        let mut above = std::ptr::null_mut();
        let mut below = std::ptr::null_mut();
        let mut beyond = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_diag(matrix, 0, &mut main), OK);
        assert_eq!(super::super::native_support::ints(main).unwrap(), [1, 4]);
        assert_eq!(terlan_ndarray_v1_diag(matrix, 1, &mut above), OK);
        assert_eq!(super::super::native_support::ints(above).unwrap(), [2]);
        assert_eq!(terlan_ndarray_v1_diag(matrix, -2, &mut below), OK);
        assert_eq!(super::super::native_support::ints(below).unwrap(), [5]);
        assert_eq!(terlan_ndarray_v1_diag(matrix, -3, &mut beyond), OK);
        assert_eq!(shape(beyond).unwrap(), [0]);

        let mut column_major = std::ptr::null_mut();
        let mut column_major_diagonal = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_eye(4, 3, -1, 2, 1, &mut column_major),
            OK
        );
        assert_eq!(
            terlan_ndarray_v1_diag(column_major, -1, &mut column_major_diagonal),
            OK
        );
        assert_eq!(
            super::super::native_support::ints(column_major_diagonal).unwrap(),
            [1, 1, 1]
        );

        let cube = array(&[1.0; 8], &[2, 2, 2]);
        let mut rejected = vector;
        assert_eq!(
            terlan_ndarray_v1_diag(cube, 0, &mut rejected),
            SHAPE_MISMATCH
        );
        assert!(rejected.is_null());
        assert_eq!(
            terlan_ndarray_v1_diag(vector, i64::MIN, &mut rejected),
            INVALID_ARGUMENT
        );
        assert!(rejected.is_null());
        assert_eq!(
            terlan_ndarray_v1_diag(vector, 0, std::ptr::null_mut()),
            INVALID_ARGUMENT
        );

        super::super::native_support::delete_arrays(&[
            vector,
            centered,
            upper,
            lower,
            matrix,
            main,
            above,
            below,
            beyond,
            column_major,
            column_major_diagonal,
            cube,
        ]);
    }
}

#[test]
fn broadcasting_mesh_and_triangle_indexes_have_stable_owned_results() {
    unsafe {
        let x = array(&[1.0, 2.0], &[2]);
        let y = array(&[10.0, 20.0, 30.0], &[3]);
        let peers = [y as usize as u64];
        let mut outputs = std::ptr::null_mut();
        let mut length = 0usize;
        assert_eq!(
            terlan_ndarray_v1_meshgrid(
                x,
                peers.as_ptr(),
                1,
                0,
                false,
                true,
                &mut outputs,
                &mut length,
            ),
            0
        );
        assert_eq!(length, 2);
        let grids = std::slice::from_raw_parts(outputs, length);
        assert_eq!(shape(grids[0] as usize as Array).unwrap(), [3, 2]);
        for &grid in grids {
            super::super::native_support::delete_arrays(&[grid as usize as Array]);
        }
        crate::ffi::terlan_ndarray_v1_delete_handle_values(outputs);
        let mut triangle = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_tril_indices(3, 0, 3, &mut triangle), 0);
        assert_eq!(shape(triangle).unwrap(), [2, 6]);
        super::super::native_support::delete_arrays(&[x, y, triangle]);
    }
}

#[test]
fn meshgrid_matches_xy_ij_sparse_copy_dtype_and_rank_semantics() {
    unsafe {
        let x_values = [1.0, 2.0];
        let x_shape = [2i64];
        let mut x = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_from_floats_as(
                x_values.as_ptr(),
                2,
                x_shape.as_ptr(),
                1,
                11,
                &mut x,
            ),
            OK
        );
        let y = array(&[10.0, 20.0, 30.0], &[3]);
        let z = array(&[5.0, 6.0], &[2]);
        let scalar = array(&[7.0], &[]);
        let peers = [y as usize as u64];
        let peers_3d = [y as usize as u64, z as usize as u64];
        let mut raw = std::ptr::null_mut();
        let mut length = 0usize;

        assert_eq!(
            terlan_ndarray_v1_meshgrid(x, peers.as_ptr(), 1, 0, false, true, &mut raw, &mut length,),
            OK
        );
        let xy = take_arrays(raw, length);
        assert_eq!(shape(xy[0]).unwrap(), [3, 2]);
        assert_eq!(values(xy[0]), [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        assert_eq!(values(xy[1]), [10.0, 10.0, 20.0, 20.0, 30.0, 30.0]);
        assert_eq!(super::super::native_support::dtype(xy[0]).unwrap(), 11);
        assert_eq!(super::super::native_support::dtype(xy[1]).unwrap(), 3);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_float(xy[1], &[0i64, 0] as *const i64, 2, 9.0),
            OK
        );
        assert_eq!(values(xy[1]), [9.0, 10.0, 20.0, 20.0, 30.0, 30.0]);
        assert_eq!(values(y), [10.0, 20.0, 30.0]);

        raw = std::ptr::null_mut();
        length = 0;
        assert_eq!(
            terlan_ndarray_v1_meshgrid(
                x,
                peers.as_ptr(),
                1,
                1,
                false,
                false,
                &mut raw,
                &mut length,
            ),
            OK
        );
        let ij_views = take_arrays(raw, length);
        assert_eq!(shape(ij_views[0]).unwrap(), [2, 3]);
        assert_eq!(values(ij_views[0]), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        assert_eq!(values(ij_views[1]), [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
        let mut shares = false;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_shares_memory(x, ij_views[0], &mut shares),
            OK
        );
        assert!(shares);

        raw = std::ptr::null_mut();
        length = 0;
        assert_eq!(
            terlan_ndarray_v1_meshgrid(x, peers.as_ptr(), 1, 0, true, true, &mut raw, &mut length,),
            OK
        );
        let sparse = take_arrays(raw, length);
        assert_eq!(shape(sparse[0]).unwrap(), [1, 2]);
        assert_eq!(shape(sparse[1]).unwrap(), [3, 1]);

        raw = std::ptr::null_mut();
        length = 0;
        assert_eq!(
            terlan_ndarray_v1_meshgrid(
                x,
                peers_3d.as_ptr(),
                2,
                1,
                false,
                true,
                &mut raw,
                &mut length,
            ),
            OK
        );
        let grids_3d = take_arrays(raw, length);
        assert_eq!(shape(grids_3d[0]).unwrap(), [2, 3, 2]);
        assert_eq!(shape(grids_3d[1]).unwrap(), [2, 3, 2]);
        assert_eq!(shape(grids_3d[2]).unwrap(), [2, 3, 2]);

        raw = std::ptr::null_mut();
        length = 0;
        assert_eq!(
            terlan_ndarray_v1_meshgrid(
                x,
                std::ptr::null(),
                0,
                0,
                false,
                true,
                &mut raw,
                &mut length,
            ),
            OK
        );
        let single = take_arrays(raw, length);
        assert_eq!(shape(single[0]).unwrap(), [2]);
        assert_eq!(values(single[0]), [1.0, 2.0]);

        raw = 1usize as *mut u64;
        length = 99;
        assert_eq!(
            terlan_ndarray_v1_meshgrid(x, peers.as_ptr(), 1, 2, false, true, &mut raw, &mut length,),
            INVALID_ARGUMENT
        );
        assert!(raw.is_null());
        assert_eq!(length, 0);
        raw = std::ptr::null_mut();
        length = 0;
        assert_eq!(
            terlan_ndarray_v1_meshgrid(
                scalar,
                std::ptr::null(),
                0,
                0,
                false,
                true,
                &mut raw,
                &mut length,
            ),
            SHAPE_MISMATCH
        );
        assert!(raw.is_null());
        assert_eq!(length, 0);

        let mut owned = vec![x, y, z, scalar];
        owned.extend(xy);
        owned.extend(ij_views);
        owned.extend(sparse);
        owned.extend(grids_3d);
        owned.extend(single);
        super::super::native_support::delete_arrays(&owned);
    }
}

#[test]
fn shape_completion_rejects_bad_ranks_shapes_indexes_and_nulls() {
    unsafe {
        let scalar = array(&[1.0], &[]);
        let vector = array(&[1.0, 2.0], &[2]);
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_diag(scalar, 0, &mut output),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_delete_indexes(vector, &[9i64] as *const i64, 1, &mut output),
            AXIS_OUT_OF_RANGE
        );
        assert_eq!(
            terlan_ndarray_v1_full(std::ptr::null(), 1, 1.0, &mut output),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        super::super::native_support::delete_arrays(&[scalar, vector]);
    }
}

#[test]
fn sharing_queries_distinguish_storage_bounds_from_exact_overlap() {
    unsafe {
        let source = array(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[6]);
        let independent = array(&[0.0, 1.0, 2.0], &[3]);
        let mut even = std::ptr::null_mut();
        let mut odd = std::ptr::null_mut();
        let mut overlap = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(source, 0, 0, 6, 2, &mut even),
            0
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(source, 0, 1, 6, 2, &mut odd),
            0
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(source, 0, 2, 5, 1, &mut overlap),
            0
        );

        let mut answer = false;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_may_share_memory(even, odd, &mut answer),
            0
        );
        assert!(answer);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_shares_memory(even, odd, &mut answer),
            0
        );
        assert!(!answer);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_shares_memory(even, overlap, &mut answer),
            0
        );
        assert!(answer);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_may_share_memory(source, independent, &mut answer),
            0
        );
        assert!(!answer);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_shares_memory(source, overlap, std::ptr::null_mut(),),
            INVALID_ARGUMENT
        );

        super::super::native_support::delete_arrays(&[source, independent, even, odd, overlap]);
    }
}

#[test]
fn copyto_broadcasts_and_place_and_putmask_use_distinct_value_cycles() {
    unsafe {
        let destination = array(&[0.0; 6], &[2, 3]);
        let source = array(&[1.0, 2.0], &[2, 1]);
        assert_eq!(crate::ffi::terlan_ndarray_v1_copyto(destination, source), 0);
        assert_eq!(values(destination), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let placed = array(&[0.0; 5], &[5]);
        let masked = array(&[0.0; 5], &[5]);
        let mask = super::super::native_support::owned_bool(&[0, 1, 1, 0, 1], &[5]).unwrap();
        let replacements = array(&[7.0, 8.0], &[2]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_place(placed, mask, replacements),
            0
        );
        assert_eq!(values(placed), [0.0, 7.0, 8.0, 0.0, 7.0]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_putmask(masked, mask, replacements),
            0
        );
        assert_eq!(values(masked), [0.0, 8.0, 7.0, 0.0, 7.0]);

        let aliased = array(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let alias_mask = super::super::native_support::owned_bool(&[0, 1, 1, 1], &[4]).unwrap();
        let mut alias_values = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(aliased, 0, 0, 2, 1, &mut alias_values,),
            0
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_place(aliased, alias_mask, alias_values),
            0
        );
        assert_eq!(values(aliased), [1.0, 1.0, 2.0, 1.0]);

        let wrong_shape = array(&[1.0, 2.0], &[2]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto(destination, wrong_shape),
            SHAPE_MISMATCH
        );
        super::super::native_support::delete_arrays(&[
            destination,
            source,
            placed,
            masked,
            mask,
            replacements,
            aliased,
            alias_mask,
            alias_values,
            wrong_shape,
        ]);
    }
}

#[test]
fn copyto_supports_scalars_masks_casting_strides_and_overlap() {
    unsafe {
        let destination = super::super::native_support::owned_real(&[0.0; 6], &[2, 3], 11).unwrap();
        let scalar = array(&[2.5], &[]);
        assert_eq!(crate::ffi::terlan_ndarray_v1_copyto(destination, scalar), 0);
        assert_eq!(values(destination), [2.5; 6]);

        let mut transposed = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_transpose(destination, 0, 1, &mut transposed),
            0
        );
        let scalar_three = array(&[3.5], &[]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto(transposed, scalar_three),
            0
        );
        assert_eq!(values(destination), [3.5; 6]);

        let mask = super::super::native_support::owned_bool(&[0, 1, 0], &[3]).unwrap();
        let scalar_four = array(&[4.5], &[]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto_where(destination, scalar_four, mask, 3),
            0
        );
        assert_eq!(values(destination), [3.5, 4.5, 3.5, 3.5, 4.5, 3.5]);

        let integers = super::super::native_support::owned_int_as(&[0, 0, 0, 0], &[4], 6).unwrap();
        let fractional = array(&[1.5], &[]);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto(integers, fractional),
            DTYPE_MISMATCH
        );
        assert_eq!(
            super::super::native_support::ints(integers).unwrap(),
            [0; 4]
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto_casting(integers, fractional, 4),
            0
        );
        assert_eq!(
            super::super::native_support::ints(integers).unwrap(),
            [1; 4]
        );

        let overlap = super::super::native_support::owned_int_as(&[1, 2, 3, 4], &[4], 6).unwrap();
        let mut overlap_destination = std::ptr::null_mut();
        let mut overlap_source = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(overlap, 0, 0, 3, 1, &mut overlap_destination,),
            0
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(overlap, 0, 1, 4, 1, &mut overlap_source,),
            0
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto(overlap_destination, overlap_source),
            0
        );
        assert_eq!(
            super::super::native_support::ints(overlap).unwrap(),
            [2, 3, 4, 4]
        );

        let false_mask = super::super::native_support::owned_bool(&[0], &[]).unwrap();
        let zero = super::super::native_support::owned_int_as(&[0], &[], 6).unwrap();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copyto_where(overlap, zero, false_mask, 3),
            0
        );
        assert_eq!(
            super::super::native_support::ints(overlap).unwrap(),
            [2, 3, 4, 4]
        );

        super::super::native_support::delete_arrays(&[
            destination,
            scalar,
            transposed,
            scalar_three,
            mask,
            scalar_four,
            integers,
            fractional,
            overlap,
            overlap_destination,
            overlap_source,
            false_mask,
            zero,
        ]);
    }
}

#[test]
fn copyto_exhausts_nine_bit_masks_forward_reverse_and_strided() {
    unsafe {
        const LENGTH: usize = 9;
        let ones = array(&[1.0; LENGTH], &[LENGTH as i64]);
        for bits in 0..(1usize << LENGTH) {
            let mask_values = (0..LENGTH)
                .map(|index| u8::from(bits & (1usize << index) != 0))
                .collect::<Vec<_>>();
            let mask =
                super::super::native_support::owned_bool(&mask_values, &[LENGTH as i64]).unwrap();
            let forward = array(&[0.0; LENGTH], &[LENGTH as i64]);
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_copyto_where(forward, ones, mask, 3),
                0
            );
            assert_eq!(
                values(forward),
                mask_values
                    .iter()
                    .map(|&value| f64::from(value))
                    .collect::<Vec<_>>()
            );

            let mut reversed_mask = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_slice_axis(
                    mask,
                    0,
                    i64::MIN,
                    i64::MIN,
                    -1,
                    &mut reversed_mask,
                ),
                0
            );
            let reverse = array(&[0.0; LENGTH], &[LENGTH as i64]);
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_copyto_where(reverse, ones, reversed_mask, 3),
                0
            );
            assert_eq!(
                values(reverse),
                mask_values
                    .iter()
                    .rev()
                    .map(|&value| f64::from(value))
                    .collect::<Vec<_>>()
            );

            let strided = array(&[0.0; LENGTH], &[LENGTH as i64]);
            let mut strided_destination = std::ptr::null_mut();
            let mut strided_source = std::ptr::null_mut();
            let mut strided_mask = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_slice_axis(
                    strided,
                    0,
                    0,
                    LENGTH as i64,
                    2,
                    &mut strided_destination,
                ),
                0
            );
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_slice_axis(
                    ones,
                    0,
                    0,
                    LENGTH as i64,
                    2,
                    &mut strided_source,
                ),
                0
            );
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_slice_axis(
                    mask,
                    0,
                    0,
                    LENGTH as i64,
                    2,
                    &mut strided_mask,
                ),
                0
            );
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_copyto_where(
                    strided_destination,
                    strided_source,
                    strided_mask,
                    3,
                ),
                0
            );
            let expected = (0..LENGTH)
                .map(|index| {
                    if index % 2 == 0 && mask_values[index] != 0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(values(strided), expected);

            let reverse_strided = array(&[0.0; LENGTH], &[LENGTH as i64]);
            let mut reverse_destination = std::ptr::null_mut();
            let mut reverse_stride_mask = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_slice_axis(
                    reverse_strided,
                    0,
                    0,
                    LENGTH as i64,
                    2,
                    &mut reverse_destination,
                ),
                0
            );
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_slice_axis(
                    mask,
                    0,
                    i64::MIN,
                    i64::MIN,
                    -2,
                    &mut reverse_stride_mask,
                ),
                0
            );
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_copyto_where(
                    reverse_destination,
                    strided_source,
                    reverse_stride_mask,
                    3,
                ),
                0
            );
            let reverse_expected = (0..LENGTH)
                .map(|index| {
                    if index % 2 == 0 && mask_values[LENGTH - index - 1] != 0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(values(reverse_strided), reverse_expected);

            super::super::native_support::delete_arrays(&[
                mask,
                forward,
                reversed_mask,
                reverse,
                strided,
                strided_destination,
                strided_source,
                strided_mask,
                reverse_strided,
                reverse_destination,
                reverse_stride_mask,
            ]);
        }
        super::super::native_support::delete_arrays(&[ones]);
    }
}
