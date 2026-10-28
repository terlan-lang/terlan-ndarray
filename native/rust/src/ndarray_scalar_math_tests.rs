//! NumPy `test_scalarmath` conformance for admitted homogeneous scalar-array behavior.

use rustfft::num_complex::Complex64;

use super::native_support::{
    delete_arrays, dtype, floats, ints, numeric, owned_bool, owned_complex, owned_int_as,
    owned_real, owned_uint, uints, Array, DOMAIN_ERROR, OK, UNSUPPORTED_DTYPE,
};

const DIVIDE_BY_ZERO: i32 = 125;

/// Creates one rank-zero value in a canonical ABI dtype.
unsafe fn scalar(dtype_code: i32, real: f64, imag: f64) -> Array {
    match dtype_code {
        1 => unsafe { owned_bool(&[u8::from(real != 0.0 || imag != 0.0)], &[]).unwrap() },
        2 | 4..=9 => unsafe { owned_int_as(&[real as i64], &[], dtype_code).unwrap() },
        10 => unsafe { owned_uint(&[real as u64], &[]).unwrap() },
        3 | 11 | 14 => unsafe { owned_real(&[real], &[], dtype_code).unwrap() },
        12 | 13 => unsafe {
            owned_complex(&[Complex64::new(real, imag)], &[], dtype_code).unwrap()
        },
        _ => panic!("unsupported fixture dtype {dtype_code}"),
    }
}

/// Runs one binary array operation and returns its owned output.
unsafe fn binary(
    operation: unsafe extern "C" fn(Array, Array, *mut Array) -> i32,
    lhs: Array,
    rhs: Array,
) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(unsafe { operation(lhs, rhs, &mut output) }, OK);
    assert!(!output.is_null());
    output
}

/// Runs one unary array operation and returns its owned output.
unsafe fn unary(operation: unsafe extern "C" fn(Array, *mut Array) -> i32, input: Array) -> Array {
    let mut output = std::ptr::null_mut();
    assert_eq!(unsafe { operation(input, &mut output) }, OK);
    assert!(!output.is_null());
    output
}

/// Returns the storage width of one admitted integer dtype.
fn integer_bits(dtype_code: i32) -> u32 {
    match dtype_code {
        4 | 7 => 8,
        5 | 8 => 16,
        6 | 9 => 32,
        2 | 10 => 64,
        _ => panic!("not an integer dtype: {dtype_code}"),
    }
}

/// Returns whether one admitted integer dtype stores signed values.
fn integer_signed(dtype_code: i32) -> bool {
    matches!(dtype_code, 2 | 4 | 5 | 6)
}

#[test]
fn scalar_math_creation_promotion_and_relations_match_numpy_dtypes() {
    const PROMOTION: [[i32; 14]; 14] = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 3, 13, 13, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 13, 13, 3],
        [4, 2, 3, 4, 5, 6, 5, 6, 2, 3, 11, 12, 13, 14],
        [5, 2, 3, 5, 5, 6, 5, 6, 2, 3, 11, 12, 13, 11],
        [6, 2, 3, 6, 6, 6, 6, 6, 2, 3, 3, 13, 13, 3],
        [7, 2, 3, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [8, 2, 3, 6, 6, 6, 8, 8, 9, 10, 11, 12, 13, 11],
        [9, 2, 3, 2, 2, 2, 9, 9, 9, 10, 3, 13, 13, 3],
        [10, 3, 3, 3, 3, 3, 10, 10, 10, 10, 3, 13, 13, 3],
        [11, 3, 3, 11, 11, 3, 11, 11, 3, 3, 11, 12, 13, 11],
        [12, 13, 13, 12, 12, 13, 12, 12, 13, 13, 12, 12, 13, 12],
        [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13],
        [14, 3, 3, 14, 11, 3, 14, 11, 3, 3, 11, 12, 13, 14],
    ];
    unsafe {
        for dtype_code in 1..=14 {
            let value = scalar(dtype_code, 1.0, 0.0);
            assert_eq!(dtype(value).unwrap(), dtype_code);
            assert_eq!(numeric(value, false).unwrap().values[0].re, 1.0);
            delete_arrays(&[value]);
        }

        for lhs_dtype in 1..=14 {
            for rhs_dtype in 1..=14 {
                let lhs = scalar(lhs_dtype, 3.0, 0.0);
                let rhs = scalar(rhs_dtype, 1.0, 0.0);
                let sum = binary(crate::ffi::terlan_ndarray_v1_add, lhs, rhs);
                assert_eq!(
                    dtype(sum).unwrap(),
                    PROMOTION[(lhs_dtype - 1) as usize][(rhs_dtype - 1) as usize]
                );
                delete_arrays(&[sum, rhs, lhs]);
            }
        }

        for unsigned_dtype in [7, 8, 9, 10] {
            for signed_dtype in [4, 5, 6, 2] {
                let unsigned = scalar(unsigned_dtype, 1.0, 0.0);
                let signed = scalar(signed_dtype, -1.0, 0.0);
                let greater = binary(crate::ffi::terlan_ndarray_v1_greater, unsigned, signed);
                let unequal = binary(crate::ffi::terlan_ndarray_v1_not_equal, unsigned, signed);
                assert_eq!(ints(greater).unwrap(), [1]);
                assert_eq!(ints(unequal).unwrap(), [1]);
                delete_arrays(&[unequal, greater, signed, unsigned]);
            }
        }

        for lhs_dtype in [4, 5, 6, 2, 7, 8, 9, 10] {
            for rhs_dtype in [4, 5, 6, 2, 7, 8, 9, 10] {
                let lhs = scalar(lhs_dtype, 1.0, 0.0);
                let rhs = scalar(rhs_dtype, 0.0, 0.0);
                let greater = binary(crate::ffi::terlan_ndarray_v1_greater, lhs, rhs);
                let less = binary(crate::ffi::terlan_ndarray_v1_less, lhs, rhs);
                assert_eq!(ints(greater).unwrap(), [1]);
                assert_eq!(ints(less).unwrap(), [0]);
                delete_arrays(&[less, greater, rhs, lhs]);
            }
        }

        for lhs_dtype in [4, 5, 6, 2, 11, 3, 14] {
            for rhs_dtype in [4, 5, 6, 2, 11, 3, 14] {
                let lhs = scalar(lhs_dtype, 1.0, 0.0);
                let rhs = scalar(rhs_dtype, -1.0, 0.0);
                let greater = binary(crate::ffi::terlan_ndarray_v1_greater, lhs, rhs);
                let equal = binary(crate::ffi::terlan_ndarray_v1_equal, rhs, rhs);
                assert_eq!(ints(greater).unwrap(), [1]);
                assert_eq!(ints(equal).unwrap(), [1]);
                delete_arrays(&[equal, greater, rhs, lhs]);
            }
        }
    }
}

#[test]
fn scalar_math_negation_subtraction_and_negative_power_match_numpy_errors() {
    unsafe {
        let boolean = scalar(1, 1.0, 0.0);
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_negative(boolean, &mut rejected),
            UNSUPPORTED_DTYPE
        );
        assert!(rejected.is_null());
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_positive(boolean, &mut rejected),
            UNSUPPORTED_DTYPE
        );
        assert!(rejected.is_null());
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_subtract(boolean, boolean, &mut rejected),
            UNSUPPORTED_DTYPE
        );
        assert!(rejected.is_null());

        for dtype_code in 2..=14 {
            let input = scalar(
                dtype_code,
                1.0,
                if dtype_code >= 12 && dtype_code <= 13 {
                    2.0
                } else {
                    0.0
                },
            );
            let negated = unary(crate::ffi::terlan_ndarray_v1_negative, input);
            let positive = unary(crate::ffi::terlan_ndarray_v1_positive, input);
            assert_eq!(
                numeric(positive, false).unwrap().values,
                numeric(input, false).unwrap().values
            );
            let restored = binary(crate::ffi::terlan_ndarray_v1_add, negated, input);
            let values = numeric(restored, false).unwrap().values;
            assert_eq!(values, [Complex64::new(0.0, 0.0)]);
            let difference = binary(crate::ffi::terlan_ndarray_v1_subtract, input, input);
            assert_eq!(numeric(difference, false).unwrap().values, values);
            delete_arrays(&[difference, restored, positive, negated, input]);
        }

        for base_dtype in [2, 4, 5, 6, 7, 8, 9] {
            for exponent_dtype in [2, 4, 5, 6] {
                let base = scalar(base_dtype, 2.0, 0.0);
                let exponent = scalar(exponent_dtype, -1.0, 0.0);
                assert_eq!(
                    crate::ffi::terlan_ndarray_v1_power(base, exponent, &mut rejected),
                    DOMAIN_ERROR
                );
                assert!(rejected.is_null());
                delete_arrays(&[exponent, base]);
            }
        }
        let unsigned = scalar(10, 2.0, 0.0);
        let exponent = scalar(2, -1.0, 0.0);
        let reciprocal = binary(crate::ffi::terlan_ndarray_v1_power, unsigned, exponent);
        assert_eq!(dtype(reciprocal).unwrap(), 3);
        assert_eq!(floats(reciprocal, false).unwrap(), [0.5]);
        delete_arrays(&[reciprocal, exponent, unsigned, boolean]);
    }
}

#[test]
fn scalar_math_floor_divide_and_remainder_match_numpy_sign_and_zero_rules() {
    unsafe {
        for lhs_dtype in [2, 4, 5, 6, 7, 8, 9, 10] {
            for rhs_dtype in [2, 4, 5, 6, 7, 8, 9, 10] {
                for lhs_sign in if integer_signed(lhs_dtype) {
                    [-1, 1]
                } else {
                    [1, 1]
                } {
                    for rhs_sign in if integer_signed(rhs_dtype) {
                        [-1, 1]
                    } else {
                        [1, 1]
                    } {
                        let lhs = scalar(lhs_dtype, f64::from(lhs_sign * 71), 0.0);
                        let rhs = scalar(rhs_dtype, f64::from(rhs_sign * 19), 0.0);
                        let quotient = binary(crate::ffi::terlan_ndarray_v1_floor_divide, lhs, rhs);
                        let remainder = binary(crate::ffi::terlan_ndarray_v1_remainder, lhs, rhs);
                        let q = numeric(quotient, false).unwrap().values[0].re;
                        let r = numeric(remainder, false).unwrap().values[0].re;
                        let a = f64::from(lhs_sign * 71);
                        let b = f64::from(rhs_sign * 19);
                        assert_eq!(q * b + r, a);
                        assert!(if rhs_sign < 0 {
                            b < r && r <= 0.0
                        } else {
                            0.0 <= r && r < b
                        });
                        delete_arrays(&[remainder, quotient, rhs, lhs]);
                    }
                }
            }
        }

        let mut dividends = Vec::new();
        let mut divisors = Vec::new();
        for dividend in -127..=127 {
            for divisor in -127..=127 {
                if divisor != 0 {
                    dividends.push(f64::from(dividend));
                    divisors.push(f64::from(divisor));
                }
            }
        }
        let shape = [dividends.len() as i64];
        for dtype_code in [14, 11, 3] {
            let lhs = owned_real(&dividends, &shape, dtype_code).unwrap();
            let rhs = owned_real(&divisors, &shape, dtype_code).unwrap();
            let quotient = binary(crate::ffi::terlan_ndarray_v1_floor_divide, lhs, rhs);
            let remainder = binary(crate::ffi::terlan_ndarray_v1_remainder, lhs, rhs);
            let q = floats(quotient, false).unwrap();
            let r = floats(remainder, false).unwrap();
            let a = floats(lhs, false).unwrap();
            let b = floats(rhs, false).unwrap();
            for index in 0..a.len() {
                assert_eq!(q[index] * b[index] + r[index], a[index]);
                if r[index] == 0.0 {
                    assert_eq!(r[index].is_sign_negative(), b[index].is_sign_negative());
                }
            }
            delete_arrays(&[remainder, quotient, rhs, lhs]);
        }

        for dtype_code in [14, 11, 3] {
            let one = scalar(dtype_code, 1.0, 0.0);
            let zero = scalar(dtype_code, 0.0, 0.0);
            let quotient = binary(crate::ffi::terlan_ndarray_v1_floor_divide, one, zero);
            let remainder = binary(crate::ffi::terlan_ndarray_v1_remainder, one, zero);
            assert!(floats(quotient, false).unwrap()[0].is_infinite());
            assert!(floats(remainder, false).unwrap()[0].is_nan());
            delete_arrays(&[remainder, quotient, zero, one]);
        }
        let one = scalar(2, 1.0, 0.0);
        let zero = scalar(2, 0.0, 0.0);
        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_floor_divide(one, zero, &mut rejected),
            DIVIDE_BY_ZERO
        );
        assert!(rejected.is_null());
        delete_arrays(&[zero, one]);
    }
}

#[test]
fn scalar_math_complex_zero_division_matches_numpy_classification() {
    unsafe {
        for dtype_code in [12, 13] {
            for (value, expect_infinite) in [
                (Complex64::new(1.0, 0.0), true),
                (Complex64::new(f64::INFINITY, f64::INFINITY), true),
                (Complex64::new(f64::INFINITY, f64::NAN), true),
                (Complex64::new(f64::NAN, f64::INFINITY), true),
                (Complex64::new(f64::NAN, f64::NAN), false),
                (Complex64::new(0.0, 0.0), false),
            ] {
                let numerator = scalar(dtype_code, value.re, value.im);
                let denominator = scalar(dtype_code, 0.0, 0.0);
                let result = binary(
                    crate::ffi::terlan_ndarray_v1_true_divide,
                    numerator,
                    denominator,
                );
                let value = numeric(result, false).unwrap().values[0];
                if expect_infinite {
                    assert!(value.re.is_infinite() || value.im.is_infinite());
                } else {
                    assert!(value.re.is_nan() || value.im.is_nan());
                }
                delete_arrays(&[result, denominator, numerator]);
            }
        }
    }
}

#[test]
fn scalar_math_full_width_shifts_match_numpy_scalar_and_array_rules() {
    unsafe {
        for dtype_code in [2, 4, 5, 6, 7, 8, 9, 10] {
            let bits = integer_bits(dtype_code);
            for input in if integer_signed(dtype_code) {
                [-5.0, 5.0]
            } else {
                [5.0, 5.0]
            } {
                for shift in [bits, bits + 4] {
                    let value = scalar(dtype_code, input, 0.0);
                    let amount = scalar(dtype_code, f64::from(shift), 0.0);
                    let left = binary(crate::ffi::terlan_ndarray_v1_left_shift, value, amount);
                    let right = binary(crate::ffi::terlan_ndarray_v1_right_shift, value, amount);
                    if integer_signed(dtype_code) {
                        assert_eq!(ints(left).unwrap(), [0]);
                        assert_eq!(ints(right).unwrap(), [if input < 0.0 { -1 } else { 0 }]);
                    } else {
                        assert_eq!(uints(left).unwrap(), [0]);
                        assert_eq!(uints(right).unwrap(), [0]);
                    }
                    delete_arrays(&[right, left, amount, value]);
                }
            }

            if bits < 64 {
                let value = scalar(dtype_code, 5.0, 0.0);
                let amount = scalar(dtype_code, f64::from(bits - 1), 0.0);
                let shifted = binary(crate::ffi::terlan_ndarray_v1_left_shift, value, amount);
                if integer_signed(dtype_code) {
                    assert_eq!(ints(shifted).unwrap(), [i64::MIN >> (64 - bits)]);
                } else {
                    assert_eq!(uints(shifted).unwrap(), [1_u64 << (bits - 1)]);
                }
                delete_arrays(&[shifted, amount, value]);
            }
        }
    }
}

#[test]
fn scalar_math_large_integer_extraction_preserves_exact_values() {
    unsafe {
        let signed_values = [1_000_000, 1_000_000_000_000, 1_000_000_000_000_000_000];
        let signed = owned_int_as(&signed_values, &[3], 2).unwrap();
        assert_eq!(ints(signed).unwrap(), signed_values);
        let unsigned_values = [1_000_000_u64, 1_000_000_000_000, 1_000_000_000_000_000_000];
        let unsigned = owned_uint(&unsigned_values, &[3]).unwrap();
        assert_eq!(uints(unsigned).unwrap(), unsigned_values);
        delete_arrays(&[unsigned, signed]);
    }
}
