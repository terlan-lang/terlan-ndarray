//! NumPy Boolean operation and floating comparison conformance.

use rustfft::num_complex::Complex64;

use super::native_support::{
    delete_arrays, dtype, ints, owned_bool, owned_complex, owned_int_as, owned_real, owned_uint,
    Array, OK,
};

/// One unary native array operation.
type UnaryOperation = unsafe extern "C" fn(Array, *mut Array) -> i32;

/// One binary native array operation.
type BinaryOperation = unsafe extern "C" fn(Array, Array, *mut Array) -> i32;

/// Executes one unary operation and returns its owned result.
unsafe fn unary(operation: UnaryOperation, array: Array) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(unsafe { operation(array, &mut output) }, OK);
    assert!(!output.is_null());
    output
}

/// Executes one binary operation and returns its owned result.
unsafe fn binary(operation: BinaryOperation, lhs: Array, rhs: Array) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(unsafe { operation(lhs, rhs, &mut output) }, OK);
    assert!(!output.is_null());
    output
}

/// Returns a one-dimensional unit-stride view beginning at one offset.
unsafe fn offset_view(array: Array, start: i64, length: i64) -> Array {
    let stop = start + length;
    let step = 1_i64;
    let mut output = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            crate::ffi::terlan_ndarray_v1_basic_slice(
                array,
                &start,
                1,
                &stop,
                1,
                &step,
                1,
                -1,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &mut output,
            )
        },
        OK
    );
    output
}

/// Asserts one canonical Bool result mask.
unsafe fn assert_mask(array: Array, expected: &[i64]) {
    assert_eq!(unsafe { dtype(array) }.unwrap(), 1);
    assert_eq!(unsafe { ints(array) }.unwrap(), expected);
}

/// Runs one binary truth operation and validates its result.
unsafe fn assert_binary_mask(operation: BinaryOperation, lhs: Array, rhs: Array, expected: &[i64]) {
    let output = unsafe { binary(operation, lhs, rhs) };
    unsafe { assert_mask(output, expected) };
    unsafe { delete_arrays(&[output]) };
}

/// Creates one rank-one truth-table operand in an admitted dtype.
unsafe fn truth_operand(values: &[i64], dtype_code: i32) -> Array {
    match dtype_code {
        1 => unsafe {
            owned_bool(
                &values
                    .iter()
                    .map(|value| u8::from(*value != 0))
                    .collect::<Vec<_>>(),
                &[values.len() as i64],
            )
            .unwrap()
        },
        2 | 4..=9 => unsafe { owned_int_as(values, &[values.len() as i64], dtype_code).unwrap() },
        10 => unsafe {
            owned_uint(
                &values.iter().map(|value| *value as u64).collect::<Vec<_>>(),
                &[values.len() as i64],
            )
            .unwrap()
        },
        3 | 11 | 14 => unsafe {
            owned_real(
                &values.iter().map(|value| *value as f64).collect::<Vec<_>>(),
                &[values.len() as i64],
                dtype_code,
            )
            .unwrap()
        },
        12 | 13 => unsafe {
            owned_complex(
                &values
                    .iter()
                    .map(|value| Complex64::new(*value as f64, 0.0))
                    .collect::<Vec<_>>(),
                &[values.len() as i64],
                dtype_code,
            )
            .unwrap()
        },
        _ => panic!("unsupported truth-table dtype {dtype_code}"),
    }
}

#[test]
fn bool_logical_bitwise_and_minmax_truth_tables_match_numpy() {
    const LHS: [i64; 4] = [0, 0, 3, 2];
    const RHS: [i64; 4] = [0, 4, 0, 2];
    const OR: [i64; 4] = [0, 1, 1, 1];
    const AND: [i64; 4] = [0, 0, 0, 1];
    const XOR: [i64; 4] = [0, 1, 1, 0];
    unsafe {
        for dtype_code in 1..=14 {
            let lhs = truth_operand(&LHS, dtype_code);
            let rhs = truth_operand(&RHS, dtype_code);
            assert_binary_mask(crate::ffi::terlan_ndarray_v1_logical_or, lhs, rhs, &OR);
            assert_binary_mask(crate::ffi::terlan_ndarray_v1_logical_and, lhs, rhs, &AND);
            assert_binary_mask(crate::ffi::terlan_ndarray_v1_logical_xor, lhs, rhs, &XOR);
            assert_binary_mask(crate::ffi::terlan_ndarray_v1_not_equal, lhs, rhs, &XOR);
            let maximum = binary(crate::ffi::terlan_ndarray_v1_elementwise_maximum, lhs, rhs);
            let minimum = binary(crate::ffi::terlan_ndarray_v1_elementwise_minimum, lhs, rhs);
            let maximum_false = unary(crate::ffi::terlan_ndarray_v1_logical_not, maximum);
            let maximum_truth = unary(crate::ffi::terlan_ndarray_v1_logical_not, maximum_false);
            let minimum_false = unary(crate::ffi::terlan_ndarray_v1_logical_not, minimum);
            let minimum_truth = unary(crate::ffi::terlan_ndarray_v1_logical_not, minimum_false);
            assert_mask(maximum_truth, &OR);
            assert_mask(minimum_truth, &AND);
            delete_arrays(&[
                minimum_truth,
                minimum_false,
                maximum_truth,
                maximum_false,
                minimum,
                maximum,
                rhs,
                lhs,
            ]);
        }

        let lhs = owned_bool(&[0, 0, 1, 1], &[4]).unwrap();
        let rhs = owned_bool(&[0, 1, 0, 1], &[4]).unwrap();
        assert_binary_mask(crate::ffi::terlan_ndarray_v1_bitwise_or, lhs, rhs, &OR);
        assert_binary_mask(crate::ffi::terlan_ndarray_v1_bitwise_and, lhs, rhs, &AND);
        assert_binary_mask(crate::ffi::terlan_ndarray_v1_bitwise_xor, lhs, rhs, &XOR);
        delete_arrays(&[rhs, lhs]);
    }
}

#[test]
fn bool_logical_absolute_and_destination_operations_match_numpy() {
    unsafe {
        let true_base = owned_bool(&[1; 41], &[41]).unwrap();
        let false_base = owned_bool(&[0; 41], &[41]).unwrap();
        let destination_base = owned_bool(&[0; 42], &[42]).unwrap();
        let truth = offset_view(true_base, 1, 40);
        let falsity = offset_view(false_base, 1, 40);
        let destination = offset_view(destination_base, 2, 40);
        let mut mostly_false = [0_u8; 40];
        mostly_false[3] = 1;
        mostly_false[38] = 1;
        let mostly_true = mostly_false.map(|value| u8::from(value == 0));
        let mixed_false = owned_bool(&mostly_false, &[40]).unwrap();
        let mixed_true = owned_bool(&mostly_true, &[40]).unwrap();
        let scalar_true = owned_bool(&[1], &[]).unwrap();
        let scalar_false = owned_bool(&[0], &[]).unwrap();
        let all_true = [1_i64; 40];
        let all_false = [0_i64; 40];
        let mixed_false_int = mostly_false.map(i64::from);
        let mixed_true_int = mostly_true.map(i64::from);

        for (source, expected) in [
            (truth, &all_false[..]),
            (falsity, &all_true[..]),
            (mixed_false, &mixed_true_int[..]),
        ] {
            let inverted = unary(crate::ffi::terlan_ndarray_v1_logical_not, source);
            let bitwise = unary(crate::ffi::terlan_ndarray_v1_invert, source);
            assert_mask(inverted, expected);
            assert_mask(bitwise, expected);
            delete_arrays(&[bitwise, inverted]);
        }
        for (source, expected) in [
            (truth, &all_true[..]),
            (falsity, &all_false[..]),
            (mixed_false, &mixed_false_int[..]),
        ] {
            let absolute = unary(crate::ffi::terlan_ndarray_v1_absolute, source);
            assert_mask(absolute, expected);
            delete_arrays(&[absolute]);
        }

        for (lhs, rhs, expected) in [
            (truth, truth, &all_true[..]),
            (falsity, falsity, &all_false[..]),
            (truth, falsity, &all_true[..]),
            (falsity, truth, &all_true[..]),
            (mixed_false, truth, &all_true[..]),
            (mixed_true, falsity, &mixed_true_int[..]),
            (mixed_false, scalar_true, &all_true[..]),
            (mixed_true, scalar_false, &mixed_true_int[..]),
        ] {
            assert_binary_mask(crate::ffi::terlan_ndarray_v1_logical_or, lhs, rhs, expected);
        }
        for (lhs, rhs, expected) in [
            (truth, truth, &all_true[..]),
            (falsity, falsity, &all_false[..]),
            (truth, falsity, &all_false[..]),
            (falsity, truth, &all_false[..]),
            (mixed_false, truth, &mixed_false_int[..]),
            (mixed_true, falsity, &all_false[..]),
            (mixed_false, scalar_true, &mixed_false_int[..]),
            (mixed_true, scalar_false, &all_false[..]),
        ] {
            assert_binary_mask(
                crate::ffi::terlan_ndarray_v1_logical_and,
                lhs,
                rhs,
                expected,
            );
        }
        for (lhs, rhs, expected) in [
            (truth, truth, &all_false[..]),
            (falsity, falsity, &all_false[..]),
            (truth, falsity, &all_true[..]),
            (falsity, truth, &all_true[..]),
            (mixed_false, truth, &mixed_true_int[..]),
            (mixed_true, falsity, &mixed_true_int[..]),
            (mixed_false, scalar_true, &mixed_true_int[..]),
            (mixed_true, scalar_false, &mixed_true_int[..]),
        ] {
            assert_binary_mask(
                crate::ffi::terlan_ndarray_v1_logical_xor,
                lhs,
                rhs,
                expected,
            );
        }

        assert_eq!(
            crate::ffi::terlan_ndarray_v1_logical_not_into(destination, truth),
            OK
        );
        assert_mask(destination, &all_false);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_absolute_into(destination, truth),
            OK
        );
        assert_mask(destination, &all_true);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_logical_or_into(destination, truth, falsity),
            OK
        );
        assert_mask(destination, &all_true);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_logical_and_into(destination, truth, falsity),
            OK
        );
        assert_mask(destination, &all_false);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_logical_xor_into(destination, truth, truth),
            OK
        );
        assert_mask(destination, &all_false);

        assert_eq!(
            crate::ffi::terlan_ndarray_v1_logical_not_into(mixed_false, mixed_false),
            OK
        );
        assert_mask(mixed_false, &mixed_true_int);
        delete_arrays(&[
            scalar_false,
            scalar_true,
            mixed_true,
            mixed_false,
            destination,
            falsity,
            truth,
            destination_base,
            false_base,
            true_base,
        ]);
    }
}

/// Builds NumPy's SIMD comparison fixture for one floating precision.
fn comparison_fixture(groups: usize, lanes: usize) -> (Vec<f64>, Vec<i64>) {
    let mut values = Vec::with_capacity(groups * lanes);
    let mut truth = Vec::with_capacity(groups * lanes);
    for group in 0..groups {
        for lane in 0..lanes {
            let value = group & (1_usize << lane);
            values.push(value as f64);
            truth.push(i64::from(value != 0));
        }
    }
    (values, truth)
}

/// Validates comparisons for every admitted source offset.
fn assert_comparisons(dtype_code: i32, groups: usize, lanes: usize, offsets: usize) {
    unsafe {
        let (values, truth) = comparison_fixture(groups, lanes);
        let array = owned_real(&values, &[values.len() as i64], dtype_code).unwrap();
        let zero = owned_real(&[0.0], &[], dtype_code).unwrap();
        let one = owned_real(&[1.0], &[], dtype_code).unwrap();
        for offset in 0..offsets {
            let view = offset_view(array, offset as i64, (values.len() - offset) as i64);
            let expected = &truth[offset..];
            let expected_inverse = expected.iter().map(|value| 1 - value).collect::<Vec<_>>();
            assert_binary_mask(crate::ffi::terlan_ndarray_v1_greater, view, zero, expected);

            let shifted = binary(crate::ffi::terlan_ndarray_v1_subtract, view, one);
            assert_binary_mask(
                crate::ffi::terlan_ndarray_v1_greater_equal,
                shifted,
                zero,
                expected,
            );
            assert_binary_mask(
                crate::ffi::terlan_ndarray_v1_equal,
                view,
                zero,
                &expected_inverse,
            );

            let negated = unary(crate::ffi::terlan_ndarray_v1_negative, view);
            assert_binary_mask(crate::ffi::terlan_ndarray_v1_less, negated, zero, expected);
            let raised = binary(crate::ffi::terlan_ndarray_v1_add, negated, one);
            assert_binary_mask(
                crate::ffi::terlan_ndarray_v1_less_equal,
                raised,
                zero,
                expected,
            );
            assert_binary_mask(
                crate::ffi::terlan_ndarray_v1_not_equal,
                view,
                zero,
                expected,
            );
            assert_binary_mask(
                crate::ffi::terlan_ndarray_v1_not_equal,
                zero,
                view,
                expected,
            );
            delete_arrays(&[raised, negated, shifted, view]);
        }
        delete_arrays(&[one, zero, array]);
    }
}

/// Validates non-finite and sign classification for every admitted offset.
fn assert_classification(dtype_code: i32, groups: usize, lanes: usize, offsets: usize) {
    unsafe {
        let (values, truth) = comparison_fixture(groups, lanes);
        let mut nan_values = values.clone();
        let mut infinite_values = values.clone();
        let mut signed_values = values.clone();
        let mut infinite_truth = truth.clone();
        for index in 0..values.len() {
            if truth[index] == 0 {
                continue;
            }
            nan_values[index] = f64::NAN;
            infinite_values[index] = match index % 3 {
                0 => f64::INFINITY,
                1 => f64::NEG_INFINITY,
                _ => {
                    infinite_truth[index] = 0;
                    f64::NAN
                }
            };
            signed_values[index] = match index % 6 {
                1 => f64::NEG_INFINITY,
                3 => -f64::NAN,
                4 => -0.0,
                _ => -signed_values[index],
            };
        }
        let nan_array = owned_real(&nan_values, &[values.len() as i64], dtype_code).unwrap();
        let infinite_array =
            owned_real(&infinite_values, &[values.len() as i64], dtype_code).unwrap();
        let signed_array = owned_real(&signed_values, &[values.len() as i64], dtype_code).unwrap();
        for offset in 0..offsets {
            let nan_view = offset_view(nan_array, offset as i64, (values.len() - offset) as i64);
            let infinite_view = offset_view(
                infinite_array,
                offset as i64,
                (values.len() - offset) as i64,
            );
            let signed_view =
                offset_view(signed_array, offset as i64, (values.len() - offset) as i64);
            let expected = &truth[offset..];
            let expected_finite = expected.iter().map(|value| 1 - value).collect::<Vec<_>>();
            let is_nan = unary(crate::ffi::terlan_ndarray_v1_is_nan, nan_view);
            let nan_finite = unary(crate::ffi::terlan_ndarray_v1_is_finite, nan_view);
            let is_finite = unary(crate::ffi::terlan_ndarray_v1_is_finite, infinite_view);
            let is_infinite = unary(crate::ffi::terlan_ndarray_v1_is_infinite, infinite_view);
            let sign = unary(crate::ffi::terlan_ndarray_v1_sign_bit, signed_view);
            assert_mask(is_nan, expected);
            assert_mask(nan_finite, &expected_finite);
            assert_mask(is_finite, &expected_finite);
            assert_mask(is_infinite, &infinite_truth[offset..]);
            assert_mask(sign, expected);
            delete_arrays(&[
                sign,
                is_infinite,
                is_finite,
                nan_finite,
                is_nan,
                signed_view,
                infinite_view,
                nan_view,
            ]);
        }
        delete_arrays(&[signed_array, infinite_array, nan_array]);
    }
}

#[test]
fn float32_and_float64_boolean_comparisons_match_numpy_offset_fixtures() {
    assert_comparisons(11, 32, 8, 4);
    assert_comparisons(3, 16, 4, 2);
    assert_classification(11, 32, 8, 4);
    assert_classification(3, 16, 4, 2);
}

#[test]
fn bool_destinations_reject_wrong_shapes_atomically() {
    unsafe {
        let source = owned_bool(&[1, 0], &[2]).unwrap();
        let destination = owned_bool(&[1], &[1]).unwrap();
        assert_ne!(
            crate::ffi::terlan_ndarray_v1_logical_not_into(destination, source),
            OK
        );
        assert_eq!(ints(destination).unwrap(), [1]);
        assert_ne!(
            crate::ffi::terlan_ndarray_v1_logical_and_into(destination, source, source),
            OK
        );
        assert_eq!(ints(destination).unwrap(), [1]);
        delete_arrays(&[destination, source]);
    }
}
