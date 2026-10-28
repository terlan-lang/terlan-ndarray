//! NumPy 1.21.5 type-checking conformance tests.

use super::*;
use rustfft::num_complex::Complex64;

/// Creates one owned real fixture in an explicit dtype.
unsafe fn real_array(values: &[f64], dimensions: &[i64], dtype: i32) -> Array {
    unsafe { super::super::native_support::owned_real(values, dimensions, dtype).unwrap() }
}

/// Creates one owned complex fixture in an explicit dtype.
unsafe fn complex_array(values: &[Complex64], dimensions: &[i64], dtype: i32) -> Array {
    unsafe { super::super::native_support::owned_complex(values, dimensions, dtype).unwrap() }
}

/// Returns one array's logical numeric values.
unsafe fn values(array: Array) -> Vec<Complex64> {
    unsafe {
        super::super::native_support::numeric(array, false)
            .unwrap()
            .values
    }
}

/// Returns one array's Bool values as integers.
unsafe fn bools(array: Array) -> Vec<i64> {
    unsafe { super::super::native_support::ints(array).unwrap() }
}

/// Deletes one array and clears its reusable handle.
unsafe fn dispose_output(output: &mut Array) {
    unsafe { delete_arrays(&[*output]) };
    *output = std::ptr::null_mut();
}

#[test]
fn real_and_imag_preserve_numpy_component_dtypes_and_values() {
    unsafe {
        for dtype_code in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 3] {
            let source = if dtype_code == 1 {
                super::super::native_support::owned_bool(&[1, 0], &[2]).unwrap()
            } else if matches!(dtype_code, 2 | 4..=10) {
                super::super::native_support::owned_int_as(&[1, 0], &[2], dtype_code).unwrap()
            } else {
                real_array(&[1.0, 0.0], &[2], dtype_code)
            };
            let mut real = std::ptr::null_mut();
            let mut imag = std::ptr::null_mut();
            assert_eq!(
                super::super::numpy_numeric_extension::terlan_ndarray_v1_real(source, &mut real),
                OK
            );
            assert_eq!(
                super::super::numpy_numeric_extension::terlan_ndarray_v1_imag(source, &mut imag),
                OK
            );
            assert_eq!(dtype(real).unwrap(), dtype_code);
            assert_eq!(dtype(imag).unwrap(), dtype_code);
            assert_eq!(
                values(real),
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
            );
            assert_eq!(values(imag), [Complex64::new(0.0, 0.0); 2]);
            delete_arrays(&[imag, real, source]);
        }
        for (dtype_code, component_dtype) in [(12, 11), (13, 3)] {
            let source = complex_array(
                &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
                &[2],
                dtype_code,
            );
            let mut real = std::ptr::null_mut();
            let mut imag = std::ptr::null_mut();
            assert_eq!(
                super::super::numpy_numeric_extension::terlan_ndarray_v1_real(source, &mut real),
                OK
            );
            assert_eq!(
                super::super::numpy_numeric_extension::terlan_ndarray_v1_imag(source, &mut imag),
                OK
            );
            assert_eq!(dtype(real).unwrap(), component_dtype);
            assert_eq!(dtype(imag).unwrap(), component_dtype);
            assert_eq!(
                values(real),
                [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)]
            );
            assert_eq!(
                values(imag),
                [Complex64::new(2.0, 0.0), Complex64::new(-4.0, 0.0)]
            );
            delete_arrays(&[imag, real, source]);
        }
    }
}

#[test]
fn finite_nan_and_infinite_classification_covers_scalars_integers_and_complex_components() {
    unsafe {
        let real = real_array(
            &[f64::NEG_INFINITY, f64::NAN, -1.0, 0.0, 1.0, f64::INFINITY],
            &[6],
            3,
        );
        let integer = super::super::native_support::owned_int(&[1], &[]).unwrap();
        let complex = complex_array(
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(0.0, f64::NAN),
                Complex64::new(f64::INFINITY, 0.0),
                Complex64::new(0.0, f64::NEG_INFINITY),
            ],
            &[5],
            13,
        );
        for (source, expected_nan, expected_infinite, expected_finite) in [
            (
                real,
                vec![0, 1, 0, 0, 0, 0],
                vec![1, 0, 0, 0, 0, 1],
                vec![0, 0, 1, 1, 1, 0],
            ),
            (integer, vec![0], vec![0], vec![1]),
            (
                complex,
                vec![0, 1, 1, 0, 0],
                vec![0, 0, 0, 1, 1],
                vec![1, 0, 0, 0, 0],
            ),
        ] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_is_nan(source, &mut output),
                OK
            );
            assert_eq!(bools(output), expected_nan);
            dispose_output(&mut output);
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_is_infinite(source, &mut output),
                OK
            );
            assert_eq!(bools(output), expected_infinite);
            dispose_output(&mut output);
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_is_finite(source, &mut output),
                OK
            );
            assert_eq!(bools(output), expected_finite);
            dispose_output(&mut output);
        }
        delete_arrays(&[complex, integer, real]);
    }
}

#[test]
fn asfarray_and_nan_to_num_cover_defaults_complex_components_and_atomic_destinations() {
    unsafe {
        for source in [
            super::super::native_support::owned_int(&[1, 2, 3], &[3]).unwrap(),
            real_array(&[1.0, 2.0, 3.0], &[3], 11),
            complex_array(&[Complex64::new(1.0, 7.0)], &[1], 13),
        ] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                super::super::numpy_numeric_extension::terlan_ndarray_v1_asfarray(
                    source,
                    &mut output
                ),
                OK
            );
            assert_eq!(dtype(output).unwrap(), 3);
            assert_eq!(values(output)[0].re, 1.0);
            delete_arrays(&[output, source]);
        }

        for (dtype_code, maximum) in [(3, f64::MAX), (11, f64::from(f32::MAX)), (14, 65_504.0)] {
            let source = real_array(
                &[f64::NEG_INFINITY, f64::NAN, f64::INFINITY],
                &[3],
                dtype_code,
            );
            let mut output = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_nan_to_num_default(source, &mut output),
                OK
            );
            assert_eq!(dtype(output).unwrap(), dtype_code);
            assert_eq!(
                values(output),
                [
                    Complex64::new(-maximum, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(maximum, 0.0),
                ]
            );
            delete_arrays(&[output, source]);
        }

        let complex = complex_array(
            &[
                Complex64::new(f64::NAN, f64::INFINITY),
                Complex64::new(f64::NEG_INFINITY, f64::NAN),
            ],
            &[2],
            13,
        );
        let mut default_complex = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_nan_to_num_default(complex, &mut default_complex),
            OK
        );
        assert_eq!(dtype(default_complex).unwrap(), 13);
        assert_eq!(
            values(default_complex),
            [
                Complex64::new(0.0, f64::MAX),
                Complex64::new(-f64::MAX, 0.0),
            ]
        );
        let destination = complex_array(&[Complex64::new(9.0, 9.0); 2], &[2], 13);
        assert_eq!(
            terlan_ndarray_v1_nan_to_num_into(destination, complex, 10.0, 20.0, 30.0),
            OK
        );
        assert_eq!(
            values(destination),
            [Complex64::new(10.0, 20.0), Complex64::new(30.0, 10.0)]
        );
        assert!(values(complex)[0].re.is_nan());
        let wrong_shape = complex_array(&[Complex64::new(7.0, 8.0)], &[1], 13);
        assert_eq!(
            terlan_ndarray_v1_nan_to_num_into(wrong_shape, complex, 10.0, 20.0, 30.0),
            SHAPE_MISMATCH
        );
        assert_eq!(values(wrong_shape), [Complex64::new(7.0, 8.0)]);
        let wrong_dtype = real_array(&[7.0, 8.0], &[2], 3);
        assert_eq!(
            terlan_ndarray_v1_nan_to_num_into(wrong_dtype, complex, 10.0, 20.0, 30.0),
            DTYPE_MISMATCH
        );
        assert_eq!(
            values(wrong_dtype),
            [Complex64::new(7.0, 0.0), Complex64::new(8.0, 0.0)]
        );
        delete_arrays(&[
            wrong_dtype,
            wrong_shape,
            destination,
            default_complex,
            complex,
        ]);
    }
}
