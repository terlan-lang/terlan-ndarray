//! Remaining frozen NumPy umath boundary contracts for homogeneous arrays.

use super::native_support::{
    Array, OK, UNSUPPORTED_DTYPE, delete_arrays, dtype, floats, ints, numeric, owned_complex,
    owned_int_as, owned_real, uints,
};
use rustfft::num_complex::Complex64;

type BinaryOperation = unsafe extern "C" fn(Array, Array, *mut Array) -> i32;
type UnaryOperation = unsafe extern "C" fn(Array, *mut Array) -> i32;

/// Executes a unary operation and returns its owned output.
unsafe fn unary(operation: UnaryOperation, input: Array) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(unsafe { operation(input, &mut output) }, OK);
    assert!(!output.is_null());
    output
}

/// Executes a binary operation and returns its owned output.
unsafe fn binary(operation: BinaryOperation, lhs: Array, rhs: Array) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(unsafe { operation(lhs, rhs, &mut output) }, OK);
    assert!(!output.is_null());
    output
}

#[test]
fn true_divide_promotes_integer_inputs_and_preserves_floating_widths() {
    unsafe {
        for dtype_code in [2, 4, 5, 6, 7, 8, 9, 10] {
            let lhs = owned_int_as(&[10], &[1], dtype_code).unwrap();
            let rhs = owned_int_as(&[20], &[1], dtype_code).unwrap();
            let quotient = binary(crate::ffi::terlan_ndarray_v1_true_divide, lhs, rhs);
            assert_eq!(dtype(quotient).unwrap(), 3);
            assert_eq!(floats(quotient, false).unwrap(), [0.5]);
            delete_arrays(&[quotient, rhs, lhs]);
        }
        for dtype_code in [14, 11, 3] {
            let lhs = owned_real(&[10.0], &[1], dtype_code).unwrap();
            let rhs = owned_real(&[20.0], &[1], dtype_code).unwrap();
            let quotient = binary(crate::ffi::terlan_ndarray_v1_true_divide, lhs, rhs);
            assert_eq!(dtype(quotient).unwrap(), dtype_code);
            assert_eq!(floats(quotient, false).unwrap(), [0.5]);
            delete_arrays(&[quotient, rhs, lhs]);
        }
        for dtype_code in [12, 13] {
            let lhs = owned_complex(&[Complex64::new(10.0, 10.0)], &[1], dtype_code).unwrap();
            let rhs = owned_complex(&[Complex64::new(20.0, 0.0)], &[1], dtype_code).unwrap();
            let quotient = binary(crate::ffi::terlan_ndarray_v1_true_divide, lhs, rhs);
            assert_eq!(dtype(quotient).unwrap(), dtype_code);
            assert_eq!(
                numeric(quotient, false).unwrap().values,
                [Complex64::new(0.5, 0.5)]
            );
            delete_arrays(&[quotient, rhs, lhs]);
        }
    }
}

#[test]
fn bitwise_operations_preserve_every_admitted_integer_width() {
    unsafe {
        for dtype_code in [2, 4, 5, 6] {
            let zero = owned_int_as(&[0], &[1], dtype_code).unwrap();
            let one = owned_int_as(&[-1], &[1], dtype_code).unwrap();
            for operation in [
                crate::ffi::terlan_ndarray_v1_bitwise_and as BinaryOperation,
                crate::ffi::terlan_ndarray_v1_bitwise_or,
                crate::ffi::terlan_ndarray_v1_bitwise_xor,
            ] {
                let output = binary(operation, zero, one);
                assert_eq!(dtype(output).unwrap(), dtype_code);
                delete_arrays(&[output]);
            }
            let inverted = unary(crate::ffi::terlan_ndarray_v1_bitwise_not, zero);
            assert_eq!(dtype(inverted).unwrap(), dtype_code);
            assert_eq!(ints(inverted).unwrap(), [-1]);
            delete_arrays(&[inverted, one, zero]);
        }
        for dtype_code in [7, 8, 9, 10] {
            let zero = owned_int_as(&[0], &[1], dtype_code).unwrap();
            let one = owned_int_as(&[1], &[1], dtype_code).unwrap();
            let output = binary(crate::ffi::terlan_ndarray_v1_bitwise_or, zero, one);
            assert_eq!(dtype(output).unwrap(), dtype_code);
            assert_eq!(uints(output).unwrap(), [1]);
            delete_arrays(&[output, one, zero]);
        }
    }
}

#[test]
fn complex_floor_division_and_remainder_are_rejected_without_outputs() {
    unsafe {
        let lhs = owned_complex(&[Complex64::new(0.9, 1.0)], &[1], 13).unwrap();
        let rhs = owned_real(&[7.0], &[], 3).unwrap();
        for operation in [
            crate::ffi::terlan_ndarray_v1_floor_divide as BinaryOperation,
            crate::ffi::terlan_ndarray_v1_remainder,
        ] {
            let mut output = std::ptr::null_mut();
            assert_eq!(operation(lhs, rhs, &mut output), UNSUPPORTED_DTYPE);
            assert!(output.is_null());
        }
        delete_arrays(&[rhs, lhs]);
    }
}

#[test]
fn floating_and_complex_power_cover_fractional_fast_and_zero_cases() {
    unsafe {
        let bases = owned_real(&[1.0, 2.0, 3.0], &[3], 3).unwrap();
        let exponent = owned_real(&[0.5], &[], 3).unwrap();
        let roots = binary(crate::ffi::terlan_ndarray_v1_power, bases, exponent);
        let root_values = floats(roots, false).unwrap();
        assert_eq!(root_values[0], 1.0);
        assert!((root_values[1] - 2.0_f64.sqrt()).abs() < 1.0e-12);
        assert!((root_values[2] - 3.0_f64.sqrt()).abs() < 1.0e-12);

        let integers = owned_int_as(&[1, 2, 3], &[3], 5).unwrap();
        let square = owned_real(&[2.0], &[], 3).unwrap();
        let fast = binary(crate::ffi::terlan_ndarray_v1_power, integers, square);
        assert_eq!(floats(fast, false).unwrap(), [1.0, 4.0, 9.0]);

        let zero = owned_complex(&[Complex64::new(0.0, 0.0)], &[1], 13).unwrap();
        let zero_power = owned_complex(&[Complex64::new(0.0, 0.0)], &[], 13).unwrap();
        let positive_power = owned_real(&[0.5], &[], 3).unwrap();
        let identity = binary(crate::ffi::terlan_ndarray_v1_power, zero, zero_power);
        let positive = binary(crate::ffi::terlan_ndarray_v1_power, zero, positive_power);
        assert_eq!(
            numeric(identity, false).unwrap().values,
            [Complex64::new(1.0, 0.0)]
        );
        assert_eq!(
            numeric(positive, false).unwrap().values,
            [Complex64::new(0.0, 0.0)]
        );
        delete_arrays(&[
            positive,
            identity,
            positive_power,
            zero_power,
            zero,
            fast,
            square,
            integers,
            roots,
            exponent,
            bases,
        ]);
    }
}

#[test]
fn tiny_complex_inverse_functions_retain_real_and_imaginary_precision() {
    unsafe {
        let tiny = owned_complex(
            &[
                Complex64::new(1.0e-12, 0.0),
                Complex64::new(0.0, 1.0e-12),
                Complex64::new(1.0e-5, 1.0e-5),
            ],
            &[3],
            13,
        )
        .unwrap();
        for operation in [
            crate::ffi::terlan_ndarray_v1_arcsin as UnaryOperation,
            crate::ffi::terlan_ndarray_v1_arctan,
            crate::ffi::terlan_ndarray_v1_arcsinh,
            crate::ffi::terlan_ndarray_v1_arctanh,
        ] {
            let output = unary(operation, tiny);
            let values = numeric(output, false).unwrap().values;
            assert!((values[0].re - 1.0e-12).abs() < 1.0e-24);
            assert!(values[0].im.abs() < 1.0e-24);
            assert!((values[1].im - 1.0e-12).abs() < 1.0e-24);
            assert!(values[1].re.abs() < 1.0e-24);
            assert!(values[2].re.is_finite() && values[2].im.is_finite());
            delete_arrays(&[output]);
        }
        delete_arrays(&[tiny]);
    }
}
