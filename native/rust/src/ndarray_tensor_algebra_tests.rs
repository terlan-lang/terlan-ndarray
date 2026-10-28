use super::*;

const OK: i32 = 0;

/// Creates one owned Float64 tensor fixture.
unsafe fn array(values: &[f64], shape: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, shape, 3).unwrap() }
}

/// Creates one owned Complex128 tensor fixture.
unsafe fn complex_array(values: &[Complex64], shape: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_complex(values, shape, 13).unwrap() }
}

/// Creates one owned Bool tensor fixture.
unsafe fn bool_array(values: &[u8], shape: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_bool(values, shape).unwrap() }
}

/// Creates one owned fixture with an explicit numerical dtype.
unsafe fn typed_array(values: &[f64], shape: &[i64], dtype: i32) -> Array {
    let values = values
        .iter()
        .map(|value| Complex64::new(*value, 0.0))
        .collect::<Vec<_>>();
    unsafe { super::super::native_support::owned_numeric_as(&values, shape, dtype).unwrap() }
}

/// Reads logical complex fixture values.
unsafe fn complex_values(array: Array) -> Vec<Complex64> {
    unsafe {
        super::super::native_support::numeric(array, false)
            .unwrap()
            .values
    }
}

/// Reads logical fixture values.
unsafe fn values(array: Array) -> Vec<f64> {
    unsafe { super::super::native_support::floats(array, false).unwrap() }
}

/// Reads a fixture shape.
unsafe fn dimensions(array: Array) -> Vec<i64> {
    unsafe { super::super::native_support::shape(array).unwrap() }
}

/// Encodes borrowed array identities for the C handle-list ABI.
fn identities(arrays: &[Array]) -> Vec<u64> {
    arrays.iter().map(|array| *array as usize as u64).collect()
}

/// Executes one equation for borrowed fixture arrays.
unsafe fn einsum(equation: &str, arrays: &[Array]) -> Result<Array, i32> {
    let equation = std::ffi::CString::new(equation).unwrap();
    let operands = identities(arrays);
    let mut output = std::ptr::null_mut();
    let status = unsafe {
        terlan_ndarray_v1_einsum(
            equation.as_ptr(),
            operands.as_ptr(),
            operands.len() as i64,
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

/// Deletes fixture arrays.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { super::super::native_support::delete_arrays(arrays) }
}

#[test]
fn tensordot_and_einsum_execute_matrix_contraction() {
    unsafe {
        let lhs = array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let rhs = array(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let lhs_axes = [1];
        let rhs_axes = [0];
        let mut contracted = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_tensordot(
                lhs,
                rhs,
                lhs_axes.as_ptr(),
                1,
                rhs_axes.as_ptr(),
                1,
                &mut contracted,
            ),
            OK
        );
        assert_eq!(dimensions(contracted), [2, 2]);
        assert_eq!(values(contracted), [58.0, 64.0, 139.0, 154.0]);

        let equation = std::ffi::CString::new("ij,jk->ik").unwrap();
        let mut summed = std::ptr::null_mut();
        let operands = identities(&[lhs, rhs]);
        assert_eq!(
            terlan_ndarray_v1_einsum(
                equation.as_ptr(),
                operands.as_ptr(),
                operands.len() as i64,
                &mut summed,
            ),
            OK
        );
        assert_eq!(dimensions(summed), [2, 2]);
        assert_eq!(values(summed), [58.0, 64.0, 139.0, 154.0]);

        let third = array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let operands = identities(&[lhs, rhs, third]);
        let equation = std::ffi::CString::new("ij,jk,kl->il").unwrap();
        let mut three_operand = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_einsum(
                equation.as_ptr(),
                operands.as_ptr(),
                operands.len() as i64,
                &mut three_operand,
            ),
            OK
        );
        assert_eq!(values(three_operand), [250.0, 372.0, 601.0, 894.0]);
        dispose(&[three_operand, third, summed, contracted, rhs, lhs]);
    }
}

#[test]
fn tensordot_zero_dimensions_and_scalar_empty_axes_match_numpy() {
    unsafe {
        let lhs = array(&[], &[3, 0]);
        let rhs = array(&[], &[0, 4]);
        let mut empty_product = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_tensordot(
                lhs,
                rhs,
                [1_i64].as_ptr(),
                1,
                [0_i64].as_ptr(),
                1,
                &mut empty_product,
            ),
            OK
        );
        assert_eq!(dimensions(empty_product), [3, 4]);
        assert_eq!(values(empty_product), [0.0; 12]);

        let scalar = array(&[1.0], &[]);
        let mut scalar_product = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_tensordot(
                scalar,
                scalar,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut scalar_product,
            ),
            OK
        );
        assert_eq!(dimensions(scalar_product), []);
        assert_eq!(values(scalar_product), [1.0]);
        dispose(&[scalar_product, scalar, empty_product, rhs, lhs]);
    }
}

#[test]
fn kron_and_cross_follow_numpy_axis_order() {
    unsafe {
        let lhs = array(&[1.0, 2.0], &[2]);
        let rhs = array(&[3.0, 4.0], &[2]);
        let mut product = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_kron(lhs, rhs, &mut product), OK);
        assert_eq!(values(product), [3.0, 4.0, 6.0, 8.0]);

        let x = array(&[1.0, 0.0, 0.0], &[3]);
        let y = array(&[0.0, 1.0, 0.0], &[3]);
        let mut crossed = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_cross(x, y, -1, &mut crossed), OK);
        assert_eq!(values(crossed), [0.0, 0.0, 1.0]);

        let x2 = array(&[1.0, 0.0], &[2]);
        let y2 = array(&[0.0, 1.0], &[2]);
        let mut scalar_cross = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_cross(x2, y2, 0, &mut scalar_cross), OK);
        assert_eq!(dimensions(scalar_cross), []);
        assert_eq!(values(scalar_cross), [1.0]);

        let mixed_two = array(&[1.0, 2.0], &[2]);
        let mixed_three = array(&[3.0, 4.0, 5.0], &[3]);
        let mut mixed_forward = std::ptr::null_mut();
        let mut mixed_reverse = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_cross(mixed_two, mixed_three, -1, &mut mixed_forward),
            OK
        );
        assert_eq!(values(mixed_forward), [10.0, -5.0, -2.0]);
        assert_eq!(
            terlan_ndarray_v1_cross(mixed_three, mixed_two, -1, &mut mixed_reverse),
            OK
        );
        assert_eq!(values(mixed_reverse), [-10.0, 5.0, 2.0]);

        let broadcast_x = array(&[1.0, 0.0, 0.0], &[1, 3]);
        let broadcast_y = array(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]);
        let mut broadcast_cross = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_cross(broadcast_x, broadcast_y, 1, &mut broadcast_cross),
            OK
        );
        assert_eq!(dimensions(broadcast_cross), [2, 3]);
        assert_eq!(values(broadcast_cross), [0.0, 0.0, 1.0, 0.0, -1.0, 0.0]);
        dispose(&[
            broadcast_cross,
            broadcast_y,
            broadcast_x,
            mixed_reverse,
            mixed_forward,
            mixed_three,
            mixed_two,
            scalar_cross,
            y2,
            x2,
            crossed,
            y,
            x,
            product,
            rhs,
            lhs,
        ]);
    }
}

#[test]
fn complex_tensor_contraction_and_einsum_preserve_complex_values() {
    unsafe {
        let lhs = complex_array(&[Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)], &[2]);
        let rhs = complex_array(&[Complex64::new(1.0, -1.0), Complex64::new(0.0, 1.0)], &[2]);
        let axes = [0];
        let mut contracted = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_tensordot(
                lhs,
                rhs,
                axes.as_ptr(),
                1,
                axes.as_ptr(),
                1,
                &mut contracted,
            ),
            OK
        );
        assert_eq!(complex_values(contracted), [Complex64::new(2.0, 2.0)]);
        let equation = std::ffi::CString::new("i,i->").unwrap();
        let mut summed = std::ptr::null_mut();
        let operands = identities(&[lhs, rhs]);
        assert_eq!(
            terlan_ndarray_v1_einsum(
                equation.as_ptr(),
                operands.as_ptr(),
                operands.len() as i64,
                &mut summed,
            ),
            OK
        );
        assert_eq!(complex_values(summed), [Complex64::new(2.0, 2.0)]);
        dispose(&[summed, contracted, rhs, lhs]);
    }
}

#[test]
fn tensor_algebra_rejects_bad_axes_shapes_and_equations_without_outputs() {
    unsafe {
        let lhs = array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let rhs = array(&[1.0, 2.0, 3.0], &[3]);
        let lhs_axes = [1];
        let rhs_axes = [0];
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_tensordot(
                lhs,
                rhs,
                lhs_axes.as_ptr(),
                1,
                rhs_axes.as_ptr(),
                1,
                &mut output,
            ),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_cross(lhs, rhs, 1, &mut output),
            SHAPE_MISMATCH
        );
        let bad_equation = std::ffi::CString::new("ij,jk->bad").unwrap();
        let operands = identities(&[lhs, lhs]);
        assert_eq!(
            terlan_ndarray_v1_einsum(
                bad_equation.as_ptr(),
                operands.as_ptr(),
                operands.len() as i64,
                &mut output,
            ),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_einsum(bad_equation.as_ptr(), std::ptr::null(), 0, &mut output,),
            INVALID_ARGUMENT
        );
        let null_operand = [0_u64];
        assert_eq!(
            terlan_ndarray_v1_einsum(bad_equation.as_ptr(), null_operand.as_ptr(), 1, &mut output,),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_tensordot(
                lhs,
                lhs,
                std::ptr::null(),
                -1,
                std::ptr::null(),
                -1,
                &mut output,
            ),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        dispose(&[rhs, lhs]);
    }
}

#[test]
fn einsum_normalizes_ellipsis_whitespace_uppercase_and_scalars() {
    unsafe {
        let tensor = array(&(0..24).map(f64::from).collect::<Vec<_>>(), &[2, 3, 4]);
        let weights = array(&[1.0, 2.0, 3.0], &[3]);
        let broadcast = einsum(" ij... , j... -> ij... ", &[tensor, weights]).unwrap();
        assert_eq!(dimensions(broadcast), [2, 3, 4]);
        assert_eq!(
            values(broadcast),
            [
                0.0, 1.0, 2.0, 3.0, 8.0, 10.0, 12.0, 14.0, 24.0, 27.0, 30.0, 33.0, 12.0, 13.0,
                14.0, 15.0, 32.0, 34.0, 36.0, 38.0, 60.0, 63.0, 66.0, 69.0,
            ]
        );

        let lhs = array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let rhs = array(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let uppercase = einsum("iZ,Zj->ij", &[lhs, rhs]).unwrap();
        assert_eq!(values(uppercase), [58.0, 64.0, 139.0, 154.0]);

        let scalar = array(&[3.0], &[]);
        let vector = array(&(0..9).map(f64::from).collect::<Vec<_>>(), &[9]);
        let scalar_sum = einsum(", i ->", &[scalar, vector]).unwrap();
        assert_eq!(dimensions(scalar_sum), []);
        assert_eq!(values(scalar_sum), [108.0]);
        dispose(&[
            scalar_sum, vector, scalar, uppercase, rhs, lhs, broadcast, weights, tensor,
        ]);
    }
}

#[test]
fn einsum_broadcasts_named_singletons_and_handles_diagonals() {
    unsafe {
        let many = array(&vec![1.0; 20], &[10, 2]);
        let singleton = array(&[1.0, 1.0], &[1, 2]);
        let reduced = einsum("ij,ij->j", &[many, singleton]).unwrap();
        assert_eq!(dimensions(reduced), [2]);
        assert_eq!(values(reduced), [10.0, 10.0]);

        let square = array(&(0..27).map(f64::from).collect::<Vec<_>>(), &[3, 3, 3]);
        let diagonal = einsum("...ii->...i", &[square]).unwrap();
        assert_eq!(dimensions(diagonal), [3, 3]);
        assert_eq!(
            values(diagonal),
            [0.0, 4.0, 8.0, 9.0, 13.0, 17.0, 18.0, 22.0, 26.0]
        );
        let triple = einsum("iii->i", &[square]).unwrap();
        assert_eq!(values(triple), [0.0, 13.0, 26.0]);
        dispose(&[triple, diagonal, square, reduced, singleton, many]);
    }
}

#[test]
fn singleton_einsum_permutations_and_diagonals_share_writable_storage() {
    unsafe {
        let base = array(&(0..9).map(f64::from).collect::<Vec<_>>(), &[3, 3]);
        let diagonal = einsum("ii->i", &[base]).unwrap();
        assert_eq!(dimensions(diagonal), [3]);
        assert_eq!(values(diagonal), [0.0, 4.0, 8.0]);
        let diagonal_index = [1_i64];
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_float(diagonal, diagonal_index.as_ptr(), 1, 99.0,),
            OK
        );
        assert_eq!(values(base), [0.0, 1.0, 2.0, 3.0, 99.0, 5.0, 6.0, 7.0, 8.0]);

        let transposed = einsum("ji", &[base]).unwrap();
        assert_eq!(dimensions(transposed), [3, 3]);
        let transposed_index = [0_i64, 1];
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_float(transposed, transposed_index.as_ptr(), 2, 42.0,),
            OK
        );
        assert_eq!(
            values(base),
            [0.0, 1.0, 2.0, 42.0, 99.0, 5.0, 6.0, 7.0, 8.0]
        );
        let passthrough = einsum("...", &[base]).unwrap();
        assert_eq!(values(passthrough), values(base));
        dispose(&[passthrough, transposed, diagonal, base]);
    }
}

#[test]
fn einsum_regressions_cover_collapsed_views_boolean_and_strided_outputs() {
    unsafe {
        let combined = array(&(0..9).map(f64::from).collect::<Vec<_>>(), &[1, 1, 3, 1, 3]);
        let mapped = einsum("bbcdc->d", &[combined]).unwrap();
        assert_eq!(values(mapped), [12.0]);

        let tensor = array(&(0..16).map(f64::from).collect::<Vec<_>>(), &[2, 2, 2, 2]);
        let traced = einsum("ijij->", &[tensor]).unwrap();
        assert_eq!(values(traced), [30.0]);
        let collapsed = einsum("aabb->ab", &[tensor]).unwrap();
        assert_eq!(dimensions(collapsed), [2, 2]);
        assert_eq!(values(collapsed), [0.0, 3.0, 12.0, 15.0]);

        let boolean = bool_array(&[1, 1], &[2, 1, 1]);
        let boolean_product = einsum("...ij,...jk->...ik", &[boolean, boolean]).unwrap();
        assert_eq!(dimensions(boolean_product), [2, 1, 1]);
        assert_eq!(values(boolean_product), [1.0, 1.0]);

        let destination_base = array(&[1.0; 10], &[10]);
        let mut destination = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(
                destination_base,
                0,
                0,
                10,
                2,
                &mut destination,
            ),
            OK
        );
        let ones = array(&[1.0; 25], &[5, 5]);
        let operands = identities(&[ones, ones, ones]);
        let equation = std::ffi::CString::new("mi,mi,mi->m").unwrap();
        assert_eq!(
            terlan_ndarray_v1_einsum_into(
                destination,
                equation.as_ptr(),
                operands.as_ptr(),
                operands.len() as i64,
            ),
            OK
        );
        assert_eq!(
            values(destination_base),
            [5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0]
        );
        dispose(&[
            ones,
            destination,
            destination_base,
            boolean_product,
            boolean,
            collapsed,
            traced,
            tensor,
            mapped,
            combined,
        ]);
    }
}

#[test]
fn einsum_executes_all_admitted_real_dtype_paths() {
    unsafe {
        for dtype in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14] {
            let vector = typed_array(&[1.0, 2.0, 3.0, 4.0], &[4], dtype);
            let scalar = typed_array(&[2.0], &[], dtype);
            let reduced = einsum("i->", &[vector]).unwrap();
            let squared = einsum("i,i->i", &[vector, vector]).unwrap();
            let scaled = einsum("i,->i", &[vector, scalar]).unwrap();
            let cubic = einsum("i,i,i->", &[vector, vector, vector]).unwrap();
            let quartic = einsum("i,i,i,i->", &[vector, vector, vector, vector]).unwrap();
            assert_eq!(super::super::native_support::dtype(reduced).unwrap(), dtype);
            assert_eq!(super::super::native_support::dtype(squared).unwrap(), dtype);
            assert_eq!(values(reduced), [10.0]);
            assert_eq!(values(squared), [1.0, 4.0, 9.0, 16.0]);
            assert_eq!(values(scaled), [2.0, 4.0, 6.0, 8.0]);
            assert_eq!(values(cubic), [100.0]);
            assert_eq!(
                values(quartic),
                [if matches!(dtype, 4 | 7) { 98.0 } else { 354.0 }]
            );
            dispose(&[quartic, cubic, scaled, squared, reduced, scalar, vector]);
        }
    }
}

#[test]
fn einsum_handles_large_strided_and_multi_operand_regressions() {
    unsafe {
        let lhs_values = (0..6).map(f64::from).collect::<Vec<_>>();
        let rhs_values = (0..(2 * 3 * 2731)).map(f64::from).collect::<Vec<_>>();
        let lhs = typed_array(&lhs_values, &[2, 3], 11);
        let rhs = typed_array(&rhs_values, &[2, 3, 2731], 5);
        let summed = einsum("cl,cpx->lpx", &[lhs, rhs]).unwrap();
        let lhs_axes = [0_i64];
        let rhs_axes = [0_i64];
        let mut contracted = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_tensordot(
                lhs,
                rhs,
                lhs_axes.as_ptr(),
                1,
                rhs_axes.as_ptr(),
                1,
                &mut contracted,
            ),
            OK
        );
        assert_eq!(dimensions(summed), [3, 3, 2731]);
        assert_eq!(values(summed), values(contracted));

        let a = array(&[1.0, 1.0], &[1, 2]);
        let b = array(&[1.0, 1.0, 1.0, 1.0], &[2, 2, 1]);
        let misc = einsum("ij...,j...->i...", &[a, b]).unwrap();
        assert_eq!(dimensions(misc), [1, 2, 1]);
        assert_eq!(values(misc), [2.0, 2.0]);

        let x = array(&[1.0, 2.0], &[2]);
        let y = array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let z = array(&(1..=8).map(f64::from).collect::<Vec<_>>(), &[4, 2]);
        let triple = einsum("x,yx,zx->xzy", &[x, y, z]).unwrap();
        assert_eq!(dimensions(triple), [2, 4, 2]);
        assert_eq!(
            values(triple),
            [
                1.0, 3.0, 3.0, 9.0, 5.0, 15.0, 7.0, 21.0, 8.0, 16.0, 16.0, 32.0, 24.0, 48.0, 32.0,
                64.0,
            ]
        );
        dispose(&[triple, z, y, x, misc, b, a, contracted, summed, rhs, lhs]);
    }
}

#[test]
fn einsum_broadcasting_dot_cases_match_explicit_contractions() {
    unsafe {
        let a = array(&(0..20).map(f64::from).collect::<Vec<_>>(), &[1, 5, 4]);
        let b = array(&(0..24).map(f64::from).collect::<Vec<_>>(), &[4, 6]);
        let c = array(&(0..30).map(f64::from).collect::<Vec<_>>(), &[5, 6]);
        let first = einsum("ijk,kl,jl", &[a, b, c]).unwrap();
        assert_eq!(dimensions(first), [1]);
        let expanded = array(&(0..40).map(f64::from).collect::<Vec<_>>(), &[2, 1, 5, 4]);
        let second = einsum("abjk,kl,jl", &[expanded, b, c]).unwrap();
        assert_eq!(dimensions(second), [2, 1]);
        assert_eq!(values(second)[0], values(first)[0]);

        let cube = array(&(0..64).map(f64::from).collect::<Vec<_>>(), &[2, 4, 8]);
        let edge = einsum("obk,ijk->ioj", &[cube, cube]).unwrap();
        assert_eq!(dimensions(edge), [2, 2, 4]);
        assert_eq!(values(edge).len(), 16);
        dispose(&[edge, cube, second, expanded, first, c, b, a]);
    }
}

#[test]
fn einsum_ordered_selects_c_fortran_and_any_layouts() {
    unsafe {
        let lhs_c = array(&[1.0; 30], &[2, 3, 5]);
        let rhs_c = array(&[1.0; 12], &[4, 3]);
        let mut lhs_f = std::ptr::null_mut();
        let mut rhs_f = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(lhs_c, 1, &mut lhs_f),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(rhs_c, 1, &mut rhs_f),
            OK
        );
        let equation = std::ffi::CString::new("...ft,mf->...mt").unwrap();
        let ordered = |operands: &[Array], order: i32| {
            let identities = identities(operands);
            let mut output = std::ptr::null_mut();
            let status = terlan_ndarray_v1_einsum_ordered(
                equation.as_ptr(),
                identities.as_ptr(),
                identities.len() as i64,
                order,
                &mut output,
            );
            (status, output)
        };
        let (status, c_order) = ordered(&[lhs_f, rhs_f], 0);
        assert_eq!(status, OK);
        let (status, f_order) = ordered(&[lhs_c, rhs_c], 1);
        assert_eq!(status, OK);
        let (status, any_fortran) = ordered(&[lhs_f, rhs_f], 3);
        assert_eq!(status, OK);
        let (status, any_mixed) = ordered(&[lhs_f, rhs_c], 3);
        assert_eq!(status, OK);
        let (status, keep_fortran) = ordered(&[lhs_f, rhs_f], 2);
        assert_eq!(status, OK);
        let mut c_flag = 0;
        let mut f_flag = 0;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_contiguous(c_order, &mut c_flag),
            OK
        );
        assert_eq!(c_flag, 1);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(f_order, &mut f_flag),
            OK
        );
        assert_eq!(f_flag, 1);
        f_flag = 0;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(any_fortran, &mut f_flag),
            OK
        );
        assert_eq!(f_flag, 1);
        c_flag = 0;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_contiguous(any_mixed, &mut c_flag),
            OK
        );
        assert_eq!(c_flag, 1);
        c_flag = 0;
        f_flag = 0;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_contiguous(keep_fortran, &mut c_flag),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(keep_fortran, &mut f_flag),
            OK
        );
        assert_eq!((c_flag, f_flag), (0, 0));
        assert_eq!(values(keep_fortran), vec![3.0; 40]);
        assert_eq!(dimensions(c_order), [2, 4, 5]);
        assert_eq!(values(c_order), vec![3.0; 40]);
        let (status, invalid) = ordered(&[lhs_c, rhs_c], 9);
        assert_eq!(status, INVALID_ARGUMENT);
        assert!(invalid.is_null());
        dispose(&[
            keep_fortran,
            any_mixed,
            any_fortran,
            f_order,
            c_order,
            rhs_f,
            lhs_f,
            rhs_c,
            lhs_c,
        ]);
    }
}

#[test]
fn einsum_into_is_atomic_and_supports_overlapping_operands() {
    unsafe {
        let lhs = array(&(0..9).map(f64::from).collect::<Vec<_>>(), &[3, 3]);
        let rhs = array(&(0..9).map(f64::from).collect::<Vec<_>>(), &[3, 3]);
        let operands = identities(&[lhs, rhs]);
        let equation = std::ffi::CString::new("ij,jk->ik").unwrap();
        assert_eq!(
            terlan_ndarray_v1_einsum_into(
                rhs,
                equation.as_ptr(),
                operands.as_ptr(),
                operands.len() as i64,
            ),
            OK
        );
        assert_eq!(
            values(rhs),
            [15.0, 18.0, 21.0, 42.0, 54.0, 66.0, 69.0, 90.0, 111.0]
        );

        let before = values(rhs);
        let malformed = std::ffi::CString::new("ij,jk->bad!").unwrap();
        assert_eq!(
            terlan_ndarray_v1_einsum_into(
                rhs,
                malformed.as_ptr(),
                operands.as_ptr(),
                operands.len() as i64,
            ),
            SHAPE_MISMATCH
        );
        assert_eq!(values(rhs), before);
        dispose(&[rhs, lhs]);
    }
}

#[test]
fn equation_normalization_rejects_malformed_and_incompatible_terms() {
    assert_eq!(parse_term("i..").unwrap_err(), SHAPE_MISMATCH);
    assert_eq!(parse_term("...i...").unwrap_err(), SHAPE_MISMATCH);
    assert_eq!(parse_term("i%").unwrap_err(), SHAPE_MISMATCH);
    assert_eq!(
        normalize_equation("ij->i->j", &[vec![2, 2]]).unwrap_err(),
        SHAPE_MISMATCH
    );
    assert_eq!(
        normalize_equation("i,j", &[vec![2]]).unwrap_err(),
        SHAPE_MISMATCH
    );
    assert_eq!(
        normalize_equation("ii->i", &[vec![2, 3]]).unwrap_err(),
        SHAPE_MISMATCH
    );
    assert_eq!(
        normalize_equation("i->ij", &[vec![2]]).unwrap_err(),
        SHAPE_MISMATCH
    );
    assert_eq!(
        normalize_equation("i->ii", &[vec![2]]).unwrap_err(),
        SHAPE_MISMATCH
    );
    assert_eq!(
        normalize_equation("i,i->i", &[vec![2], vec![3]]).unwrap_err(),
        SHAPE_MISMATCH
    );
    assert_eq!(
        normalize_equation("abcdefghijklmnopqrstuvwxyzA->", &[vec![1; 27]],).unwrap_err(),
        SHAPE_MISMATCH
    );
}
