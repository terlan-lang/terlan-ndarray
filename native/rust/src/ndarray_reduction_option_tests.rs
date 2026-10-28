//! NumPy-compatible reduction option and destination conformance.

use super::native_support::{
    Array, DOMAIN_ERROR, OK, SHAPE_MISMATCH, delete_arrays, floats, ints, numeric, owned_bool,
    owned_complex, owned_int_as, owned_real, shape,
};
use rustfft::num_complex::Complex64;

type Reduction = unsafe extern "C" fn(Array, *const i64, i64, i32, *mut Array) -> i32;

unsafe extern "C" {
    fn terlan_ndarray_v1_sum_where(
        array: Array,
        mask: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
        output: *mut Array,
    ) -> i32;
    fn terlan_ndarray_v1_sum_initial(
        array: Array,
        initial: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
        output: *mut Array,
    ) -> i32;
    fn terlan_ndarray_v1_sum_where_initial(
        array: Array,
        mask: Array,
        initial: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
        output: *mut Array,
    ) -> i32;
    fn terlan_ndarray_v1_minimum_initial(
        array: Array,
        initial: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
        output: *mut Array,
    ) -> i32;
    fn terlan_ndarray_v1_maximum_initial(
        array: Array,
        initial: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
        output: *mut Array,
    ) -> i32;
    fn terlan_ndarray_v1_maximum_where(
        array: Array,
        mask: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
        output: *mut Array,
    ) -> i32;
    fn terlan_ndarray_v1_maximum_where_initial(
        array: Array,
        mask: Array,
        initial: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
        output: *mut Array,
    ) -> i32;
    fn terlan_ndarray_v1_fmin_reduce(
        array: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
        output: *mut Array,
    ) -> i32;
    fn terlan_ndarray_v1_fmax_reduce(
        array: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
        output: *mut Array,
    ) -> i32;
    fn terlan_ndarray_v1_bitwise_and_reduce(
        array: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
        output: *mut Array,
    ) -> i32;
    fn terlan_ndarray_v1_bitwise_or_reduce(
        array: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
        output: *mut Array,
    ) -> i32;
    fn terlan_ndarray_v1_bitwise_xor_reduce(
        array: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
        output: *mut Array,
    ) -> i32;
    fn terlan_ndarray_v1_outer_into(destination: Array, lhs: Array, rhs: Array) -> i32;
    fn terlan_ndarray_v1_sum_into(
        destination: Array,
        array: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
    ) -> i32;
    fn terlan_ndarray_v1_minimum_into(
        destination: Array,
        array: Array,
        axes: *const i64,
        axis_count: i64,
        keep_dims: i32,
    ) -> i32;
}

/// Runs a reduction over all dimensions and returns its owned output.
unsafe fn reduce_all(operation: Reduction, array: Array) -> (i32, Array) {
    let mut output = std::ptr::null_mut();
    let status = unsafe { operation(array, std::ptr::null(), 0, 0, &mut output) };
    (status, output)
}

#[test]
fn sum_where_and_initial_cover_broadcast_axes_and_empty_inputs() {
    unsafe {
        let array = owned_real(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 3).unwrap();
        let mask = owned_bool(&[1, 0], &[2]).unwrap();
        let initial = owned_real(&[5.0], &[], 3).unwrap();
        let axis = [0i64];
        let mut masked = std::ptr::null_mut();
        let mut seeded = std::ptr::null_mut();
        let mut combined = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_sum_where(array, mask, std::ptr::null(), 0, 0, &mut masked),
            OK
        );
        assert_eq!(floats(masked, false).unwrap(), vec![4.0]);
        assert_eq!(
            terlan_ndarray_v1_sum_initial(array, initial, axis.as_ptr(), 1, 0, &mut seeded,),
            OK
        );
        assert_eq!(floats(seeded, false).unwrap(), vec![9.0, 11.0]);
        assert_eq!(
            terlan_ndarray_v1_sum_where_initial(
                array,
                mask,
                initial,
                axis.as_ptr(),
                1,
                0,
                &mut combined,
            ),
            OK
        );
        assert_eq!(floats(combined, false).unwrap(), vec![9.0, 5.0]);

        let empty = owned_real(&[], &[0], 3).unwrap();
        let mut empty_sum = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_sum_initial(empty, initial, std::ptr::null(), 0, 0, &mut empty_sum,),
            OK
        );
        assert_eq!(floats(empty_sum, false).unwrap(), vec![5.0]);
        delete_arrays(&[
            empty_sum, empty, combined, seeded, masked, initial, mask, array,
        ]);
    }
}

#[test]
fn extrema_initials_define_empty_and_masked_reductions() {
    unsafe {
        let empty = owned_real(&[], &[0], 3).unwrap();
        let positive = owned_real(&[f64::INFINITY], &[], 3).unwrap();
        let negative = owned_real(&[f64::NEG_INFINITY], &[], 3).unwrap();
        let mut minimum = std::ptr::null_mut();
        let mut maximum = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_minimum_initial(
                empty,
                positive,
                std::ptr::null(),
                0,
                0,
                &mut minimum,
            ),
            OK
        );
        assert_eq!(floats(minimum, false).unwrap(), vec![f64::INFINITY]);
        assert_eq!(
            terlan_ndarray_v1_maximum_initial(
                empty,
                negative,
                std::ptr::null(),
                0,
                0,
                &mut maximum,
            ),
            OK
        );
        assert_eq!(floats(maximum, false).unwrap(), vec![f64::NEG_INFINITY]);

        let array = owned_real(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3], 3).unwrap();
        let mask = owned_bool(&[1, 0, 1], &[3]).unwrap();
        let initial = owned_real(&[2.5], &[], 3).unwrap();
        let axis = [0i64];
        let mut selected = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_maximum_where_initial(
                array,
                mask,
                initial,
                axis.as_ptr(),
                1,
                0,
                &mut selected,
            ),
            OK
        );
        assert_eq!(floats(selected, false).unwrap(), vec![3.0, 2.5, 5.0]);
        delete_arrays(&[
            selected, initial, mask, array, maximum, minimum, negative, positive, empty,
        ]);
    }
}

#[test]
fn identityless_masked_extrema_still_reject_empty_lanes() {
    unsafe {
        let array = owned_real(&[1.0, 2.0, 3.0, 4.0], &[2, 2], 3).unwrap();
        let mask = owned_bool(&[1, 0], &[2]).unwrap();
        let axis = [0i64];
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_maximum_where(array, mask, axis.as_ptr(), 1, 0, &mut output,),
            DOMAIN_ERROR
        );
        assert_eq!(output, std::ptr::null_mut());
        delete_arrays(&[mask, array]);
    }
}

#[test]
fn fmin_and_fmax_reductions_ignore_only_nan_operands() {
    unsafe {
        let values = owned_real(&[f64::NAN, 4.0, 1.0, f64::NAN], &[4], 3).unwrap();
        let all_nan = owned_real(&[f64::NAN, f64::NAN], &[2], 3).unwrap();
        let (min_status, minimum) = reduce_all(terlan_ndarray_v1_fmin_reduce, values);
        let (max_status, maximum) = reduce_all(terlan_ndarray_v1_fmax_reduce, values);
        let (nan_status, nan_result) = reduce_all(terlan_ndarray_v1_fmin_reduce, all_nan);
        assert_eq!((min_status, max_status, nan_status), (OK, OK, OK));
        assert_eq!(floats(minimum, false).unwrap(), vec![1.0]);
        assert_eq!(floats(maximum, false).unwrap(), vec![4.0]);
        assert!(floats(nan_result, false).unwrap()[0].is_nan());
        delete_arrays(&[nan_result, maximum, minimum, all_nan, values]);
    }
}

#[test]
fn bitwise_reductions_preserve_dtype_width_and_empty_identities() {
    unsafe {
        let values = owned_int_as(&[-1, 6, 3], &[3], 4).unwrap();
        let empty = owned_int_as(&[], &[0], 4).unwrap();
        for (operation, expected, empty_expected) in [
            (terlan_ndarray_v1_bitwise_and_reduce as Reduction, 2, -1),
            (terlan_ndarray_v1_bitwise_or_reduce as Reduction, -1, 0),
            (terlan_ndarray_v1_bitwise_xor_reduce as Reduction, -6, 0),
        ] {
            let (status, output) = reduce_all(operation, values);
            let (empty_status, empty_output) = reduce_all(operation, empty);
            assert_eq!((status, empty_status), (OK, OK));
            assert_eq!(ints(output).unwrap(), vec![expected]);
            assert_eq!(ints(empty_output).unwrap(), vec![empty_expected]);
            assert_eq!(super::native_support::dtype(output).unwrap(), 4);
            delete_arrays(&[empty_output, output]);
        }
        delete_arrays(&[empty, values]);
    }
}

#[test]
fn outer_into_writes_exact_destinations_and_rejects_wrong_shapes_atomically() {
    unsafe {
        let lhs = owned_real(&[1.0, 2.0], &[2], 3).unwrap();
        let rhs = owned_real(&[3.0, 4.0, 5.0], &[3], 3).unwrap();
        let destination = owned_real(&[0.0; 6], &[2, 3], 3).unwrap();
        assert_eq!(terlan_ndarray_v1_outer_into(destination, lhs, rhs), OK);
        assert_eq!(
            floats(destination, false).unwrap(),
            vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        );
        assert_eq!(shape(destination).unwrap(), vec![2, 3]);

        let wrong = owned_real(&[17.0; 6], &[3, 2], 3).unwrap();
        assert_eq!(
            terlan_ndarray_v1_outer_into(wrong, lhs, rhs),
            SHAPE_MISMATCH
        );
        assert_eq!(floats(wrong, false).unwrap(), vec![17.0; 6]);
        delete_arrays(&[wrong, destination, rhs, lhs]);
    }
}

#[test]
fn invalid_initial_shapes_and_unseeded_empty_extrema_are_rejected() {
    unsafe {
        let empty = owned_real(&[], &[0], 3).unwrap();
        let vector_initial = owned_real(&[1.0, 2.0], &[2], 3).unwrap();
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_sum_initial(
                empty,
                vector_initial,
                std::ptr::null(),
                0,
                0,
                &mut output,
            ),
            SHAPE_MISMATCH
        );
        let (status, unseeded) = reduce_all(crate::ffi::terlan_ndarray_v1_minimum, empty);
        assert_eq!(status, DOMAIN_ERROR);
        assert!(unseeded.is_null());
        delete_arrays(&[vector_initial, empty]);
    }
}

#[test]
fn reduction_destinations_require_exact_shapes_and_preserve_strided_storage() {
    unsafe {
        let source =
            owned_real(&(0..24).map(f64::from).collect::<Vec<_>>(), &[2, 3, 4], 3).unwrap();
        let axis = [2_i64];
        let destination_base = owned_real(&[-1.0; 12], &[2, 6], 3).unwrap();
        let mut destination = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(
                destination_base,
                1,
                0,
                6,
                2,
                &mut destination,
            ),
            OK
        );
        assert_eq!(
            terlan_ndarray_v1_sum_into(destination, source, axis.as_ptr(), 1, 0),
            OK
        );
        assert_eq!(
            floats(destination, false).unwrap(),
            vec![6.0, 22.0, 38.0, 54.0, 70.0, 86.0]
        );
        assert_eq!(
            floats(destination_base, false).unwrap(),
            vec![
                6.0, -1.0, 22.0, -1.0, 38.0, -1.0, 54.0, -1.0, 70.0, -1.0, 86.0, -1.0
            ]
        );

        let wrong = owned_real(&[17.0; 3], &[3], 3).unwrap();
        assert_eq!(
            terlan_ndarray_v1_minimum_into(wrong, source, axis.as_ptr(), 1, 0),
            SHAPE_MISMATCH
        );
        assert_eq!(floats(wrong, false).unwrap(), vec![17.0; 3]);
        delete_arrays(&[wrong, destination, destination_base, source]);
    }
}

#[test]
fn identityless_reductions_cover_axis_orders_strides_and_zero_sized_lanes() {
    unsafe {
        let mut values = vec![1.0; 24];
        values[12] = 0.0;
        let array = owned_real(&values, &[2, 3, 4], 3).unwrap();
        for (axes, expected) in [
            (vec![0_i64, 1], vec![0.0, 1.0, 1.0, 1.0]),
            (vec![0_i64, 2], vec![0.0, 1.0, 1.0]),
            (vec![1_i64, 2], vec![1.0, 0.0]),
        ] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_minimum(
                    array,
                    axes.as_ptr(),
                    axes.len() as i64,
                    0,
                    &mut output,
                ),
                OK
            );
            assert_eq!(floats(output, false).unwrap(), expected);
            delete_arrays(&[output]);
        }

        let interleaved = owned_real(
            &(0..48)
                .map(|index| if index == 24 { 0.0 } else { 1.0 })
                .collect::<Vec<_>>(),
            &[2, 3, 8],
            3,
        )
        .unwrap();
        let mut strided = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(interleaved, 2, 0, 8, 2, &mut strided),
            OK
        );
        let mut minimum = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_minimum(strided, std::ptr::null(), 0, 0, &mut minimum,),
            OK
        );
        assert_eq!(floats(minimum, false).unwrap(), [0.0]);

        let zero_outer = owned_real(&[], &[0, 3], 3).unwrap();
        let zero_inner = owned_real(&[], &[3, 0], 3).unwrap();
        let axis = [1_i64];
        let mut empty_output = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_minimum(
                zero_outer,
                axis.as_ptr(),
                1,
                0,
                &mut empty_output,
            ),
            OK
        );
        assert_eq!(shape(empty_output).unwrap(), [0]);
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_minimum(zero_inner, axis.as_ptr(), 1, 0, &mut rejected,),
            DOMAIN_ERROR
        );
        assert!(rejected.is_null());
        delete_arrays(&[
            empty_output,
            zero_inner,
            zero_outer,
            minimum,
            strided,
            interleaved,
            array,
        ]);
    }
}

#[test]
fn complex_sums_preserve_components_across_widths_and_reverse_strides() {
    unsafe {
        let values = (1..=127)
            .map(|value| Complex64::new(f64::from(value), -f64::from(value)))
            .collect::<Vec<_>>();
        for dtype_code in [12, 13] {
            let source = owned_complex(&values, &[127], dtype_code).unwrap();
            let mut reversed = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_slice_axis(
                    source,
                    0,
                    i64::MIN,
                    i64::MIN,
                    -1,
                    &mut reversed,
                ),
                OK
            );
            for input in [source, reversed] {
                let mut total = std::ptr::null_mut();
                assert_eq!(
                    crate::ffi::terlan_ndarray_v1_sum(input, std::ptr::null(), 0, 0, &mut total,),
                    OK
                );
                assert_eq!(
                    numeric(total, false).unwrap().values,
                    [Complex64::new(8128.0, -8128.0)]
                );
                delete_arrays(&[total]);
            }
            delete_arrays(&[reversed, source]);
        }
    }
}
