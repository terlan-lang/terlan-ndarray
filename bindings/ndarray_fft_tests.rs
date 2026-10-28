use super::*;

unsafe fn real_array(values: &[f64], shape: &[i64]) -> Array {
    let mut array = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            crate::ffi::terlan_ndarray_v1_from_floats(
                values.as_ptr(),
                values.len() as i64,
                shape.as_ptr(),
                shape.len() as i64,
                &mut array,
            )
        },
        OK
    );
    array
}

unsafe fn real_array_as(values: &[f64], shape: &[i64], dtype: i32) -> Array {
    let mut array = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            crate::ffi::terlan_ndarray_v1_from_floats_as(
                values.as_ptr(),
                values.len() as i64,
                shape.as_ptr(),
                shape.len() as i64,
                dtype,
                &mut array,
            )
        },
        OK
    );
    array
}

unsafe fn complex_array(values: &[Complex64], shape: &[i64], dtype: i32) -> Array {
    let reals: Vec<f64> = values.iter().map(|value| value.re).collect();
    let imags: Vec<f64> = values.iter().map(|value| value.im).collect();
    let mut array = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            crate::ffi::terlan_ndarray_v1_from_complex(
                reals.as_ptr(),
                reals.len() as i64,
                imags.as_ptr(),
                imags.len() as i64,
                shape.as_ptr(),
                shape.len() as i64,
                dtype,
                &mut array,
            )
        },
        OK
    );
    array
}

unsafe fn dtype(array: Array) -> i32 {
    let mut value = 0;
    assert_eq!(
        unsafe { crate::ffi::terlan_ndarray_v1_dtype(array, &mut value) },
        OK
    );
    value
}

unsafe fn real_values(array: Array) -> Vec<f64> {
    unsafe { copied_floats(array).unwrap() }
}

unsafe fn complex_values(array: Array) -> Vec<Complex64> {
    unsafe { copied_complex(array).unwrap() }
}

unsafe fn dispose(arrays: &[Array]) {
    for &array in arrays {
        if !array.is_null() {
            assert_eq!(unsafe { crate::ffi::terlan_ndarray_v1_delete(array) }, OK);
        }
    }
}

fn close(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1.0e-9,
        "{actual:?} != {expected:?}"
    );
}

/// Computes the forward transform directly for an independent reference.
fn direct_fft(values: &[Complex64]) -> Vec<Complex64> {
    let length = values.len() as f64;
    (0..values.len())
        .map(|frequency| {
            values
                .iter()
                .enumerate()
                .map(|(sample, value)| {
                    let angle =
                        -2.0 * std::f64::consts::PI * frequency as f64 * sample as f64 / length;
                    *value * Complex64::from_polar(1.0, angle)
                })
                .sum()
        })
        .collect()
}

/// Asserts two complex sequences agree at the package FFT tolerance.
fn assert_complex_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        close(actual, expected);
    }
}

/// Compares every complex transform used by NumPy's layout regression.
unsafe fn assert_complex_transforms_equal(contiguous: Array, strided: Array) {
    for axis in 0..3i64 {
        for inverse in [false, true] {
            let mut expected = std::ptr::null_mut();
            let mut actual = std::ptr::null_mut();
            let expected_status = if inverse {
                unsafe { terlan_ndarray_v1_ifft(contiguous, axis, 0, &mut expected) }
            } else {
                unsafe { terlan_ndarray_v1_fft(contiguous, axis, 0, &mut expected) }
            };
            let actual_status = if inverse {
                unsafe { terlan_ndarray_v1_ifft(strided, axis, 0, &mut actual) }
            } else {
                unsafe { terlan_ndarray_v1_fft(strided, axis, 0, &mut actual) }
            };
            assert_eq!(expected_status, OK);
            assert_eq!(actual_status, OK);
            assert_complex_close(&unsafe { complex_values(actual) }, &unsafe {
                complex_values(expected)
            });
            unsafe { dispose(&[actual, expected]) };
        }
    }
    for axes in [[0i64, 1], [1, 2], [0, 2]] {
        for inverse in [false, true] {
            let mut expected = std::ptr::null_mut();
            let mut actual = std::ptr::null_mut();
            let expected_status = if inverse {
                unsafe { terlan_ndarray_v1_ifftn(contiguous, axes.as_ptr(), 2, 0, &mut expected) }
            } else {
                unsafe { terlan_ndarray_v1_fftn(contiguous, axes.as_ptr(), 2, 0, &mut expected) }
            };
            let actual_status = if inverse {
                unsafe { terlan_ndarray_v1_ifftn(strided, axes.as_ptr(), 2, 0, &mut actual) }
            } else {
                unsafe { terlan_ndarray_v1_fftn(strided, axes.as_ptr(), 2, 0, &mut actual) }
            };
            assert_eq!(expected_status, OK);
            assert_eq!(actual_status, OK);
            assert_complex_close(&unsafe { complex_values(actual) }, &unsafe {
                complex_values(expected)
            });
            unsafe { dispose(&[actual, expected]) };
        }
    }
}

#[test]
fn complex_one_and_multi_axis_transforms_round_trip() {
    unsafe {
        let source = real_array(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let mut spectrum = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_fft(source, 0, 0, &mut spectrum), OK);
        for (actual, expected) in complex_values(spectrum).into_iter().zip([
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ]) {
            close(actual, expected);
        }
        let mut restored = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_ifft(spectrum, 0, 0, &mut restored), OK);
        for (actual, expected) in complex_values(restored)
            .into_iter()
            .zip([1.0, 2.0, 3.0, 4.0])
        {
            close(actual, Complex64::new(expected, 0.0));
        }

        let matrix = real_array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let axes = [0, 1];
        let mut transformed = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_fftn(matrix, axes.as_ptr(), 2, 0, &mut transformed),
            OK
        );
        for (actual, expected) in complex_values(transformed).into_iter().zip([
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-4.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]) {
            close(actual, expected);
        }
        let mut matrix_restored = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_ifftn(transformed, axes.as_ptr(), 2, 0, &mut matrix_restored),
            OK
        );
        for (actual, expected) in complex_values(matrix_restored)
            .into_iter()
            .zip([1.0, 2.0, 3.0, 4.0])
        {
            close(actual, Complex64::new(expected, 0.0));
        }
        dispose(&[
            matrix_restored,
            transformed,
            matrix,
            restored,
            spectrum,
            source,
        ]);
    }
}

#[test]
fn real_and_hermitian_transforms_round_trip_even_and_odd_lengths() {
    unsafe {
        for values in [&[1.0, 0.0, -1.0, 0.0][..], &[1.0, 2.0, 3.0, 4.0, 5.0][..]] {
            let source = real_array(values, &[values.len() as i64]);
            let mut spectrum = std::ptr::null_mut();
            assert_eq!(terlan_ndarray_v1_rfft(source, -1, 0, &mut spectrum), OK);
            let mut restored = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_irfft(spectrum, values.len() as i64, -1, 0, &mut restored),
                OK
            );
            for (actual, expected) in real_values(restored)
                .into_iter()
                .zip(values.iter().copied())
            {
                close(Complex64::new(actual, 0.0), Complex64::new(expected, 0.0));
            }
            dispose(&[restored, spectrum, source]);
        }

        let matrix_values = [1.0, 2.0, 3.0, 4.0, -2.0, 0.5, 1.5, 3.0];
        let matrix = real_array(&matrix_values, &[2, 4]);
        let axes = [0i64, 1];
        let mut spectrum = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_rfftn(matrix, axes.as_ptr(), 2, 0, &mut spectrum),
            OK
        );
        assert_eq!(array_shape(spectrum).unwrap(), [2, 3]);
        let mut restored = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_irfftn(spectrum, 4, axes.as_ptr(), 2, 0, &mut restored),
            OK
        );
        assert_eq!(array_shape(restored).unwrap(), [2, 4]);
        for (actual, expected) in real_values(restored).into_iter().zip(matrix_values) {
            close(Complex64::new(actual, 0.0), Complex64::new(expected, 0.0));
        }
        dispose(&[restored, spectrum, matrix]);

        let spectrum = real_array(&[2.0, -1.0, 3.0, 0.5], &[4]);
        let mut compact = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_ihfft(spectrum, 0, 0, &mut compact), OK);
        let mut restored = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_hfft(compact, 4, 0, 0, &mut restored), OK);
        for (actual, expected) in real_values(restored).into_iter().zip([2.0, -1.0, 3.0, 0.5]) {
            close(Complex64::new(actual, 0.0), Complex64::new(expected, 0.0));
        }
        dispose(&[restored, compact, spectrum]);
    }
}

#[test]
fn frequency_bins_shifts_normalization_and_strided_input_match_contract() {
    unsafe {
        let mut frequencies = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_fftfreq(4, 0.5, &mut frequencies), OK);
        assert_eq!(real_values(frequencies), [0.0, 0.5, -1.0, -0.5]);
        let mut real_frequencies = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_rfftfreq(5, 2.0, &mut real_frequencies),
            OK
        );
        assert_eq!(real_values(real_frequencies), [0.0, 0.1, 0.2]);

        let source = real_array(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]);
        let axis = [0];
        let mut shifted = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_fftshift(source, axis.as_ptr(), 1, &mut shifted),
            OK
        );
        assert_eq!(real_values(shifted), [3.0, 4.0, 0.0, 1.0, 2.0]);
        let mut unshifted = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_ifftshift(shifted, axis.as_ptr(), 1, &mut unshifted),
            OK
        );
        assert_eq!(real_values(unshifted), [0.0, 1.0, 2.0, 3.0, 4.0]);

        let matrix = real_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mut transposed = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_transpose(matrix, 0, 1, &mut transposed),
            OK
        );
        let mut transformed = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_fft(transposed, 1, 1, &mut transformed),
            OK
        );
        assert_eq!(array_shape(transformed).unwrap(), [3, 2]);
        dispose(&[
            transformed,
            transposed,
            matrix,
            unshifted,
            shifted,
            source,
            real_frequencies,
            frequencies,
        ]);
    }
}

#[test]
fn fft_accepts_admitted_precisions_and_promotes_to_numpy_fft_dtypes() {
    unsafe {
        for input_dtype in [14, 11, 3] {
            let real = real_array_as(&[1.0, 2.0, 3.0, 4.0], &[4], input_dtype);
            let mut real_spectrum = std::ptr::null_mut();
            assert_eq!(terlan_ndarray_v1_rfft(real, 0, 0, &mut real_spectrum), OK);
            assert_eq!(dtype(real_spectrum), 13);
            let mut real_restored = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_irfft(real_spectrum, 4, 0, 0, &mut real_restored),
                OK
            );
            assert_eq!(dtype(real_restored), 3);
            for (actual, expected) in real_values(real_restored)
                .into_iter()
                .zip([1.0, 2.0, 3.0, 4.0])
            {
                close(Complex64::new(actual, 0.0), Complex64::new(expected, 0.0));
            }
            dispose(&[real_restored, real_spectrum, real]);
        }

        let values = [
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, -0.5),
            Complex64::new(3.0, 1.0),
            Complex64::new(4.0, -1.0),
        ];
        for input_dtype in [12, 13] {
            let complex = complex_array(&values, &[4], input_dtype);
            let mut spectrum = std::ptr::null_mut();
            assert_eq!(terlan_ndarray_v1_fft(complex, 0, 0, &mut spectrum), OK);
            assert_eq!(dtype(spectrum), 13);
            let mut restored = std::ptr::null_mut();
            assert_eq!(terlan_ndarray_v1_ifft(spectrum, 0, 0, &mut restored), OK);
            assert_eq!(dtype(restored), 13);
            assert_complex_close(&complex_values(restored), &values);
            dispose(&[restored, spectrum, complex]);
        }
    }
}

#[test]
fn fft_rejects_empty_axes_bad_normalization_duplicate_axes_and_shapes() {
    unsafe {
        let empty = real_array(&[], &[0]);
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_fft(empty, 0, 0, &mut output),
            DOMAIN_ERROR
        );
        assert!(output.is_null());
        let source = real_array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(
            terlan_ndarray_v1_fft(source, 0, 7, &mut output),
            INVALID_ARGUMENT
        );
        let duplicate_axes = [0, -2];
        assert_eq!(
            terlan_ndarray_v1_fftn(source, duplicate_axes.as_ptr(), 2, 0, &mut output),
            INVALID_ARGUMENT
        );
        let compact = real_array(&[1.0, 2.0], &[2]);
        assert_eq!(
            terlan_ndarray_v1_irfft(compact, 6, 0, 0, &mut output),
            SHAPE_MISMATCH
        );
        let complex = complex_array(&[Complex64::new(1.0, 1.0); 4], &[4], 13);
        assert_eq!(
            terlan_ndarray_v1_rfft(complex, 0, 0, &mut output),
            UNSUPPORTED_DTYPE
        );
        assert_eq!(
            terlan_ndarray_v1_ihfft(complex, 0, 0, &mut output),
            UNSUPPORTED_DTYPE
        );
        assert_eq!(
            terlan_ndarray_v1_fftfreq(0, 1.0, &mut output),
            INVALID_ARGUMENT
        );
        dispose(&[complex, compact, source, empty]);
    }
}

#[test]
fn numpy_fft_direct_values_and_all_normalizations_match() {
    unsafe {
        let values = (0..30)
            .map(|index| Complex64::new((index as f64 * 0.37).sin(), (index as f64 * 0.19).cos()))
            .collect::<Vec<_>>();
        let source = complex_array(&values, &[30], 13);
        let reference = direct_fft(&values);
        for normalization in 0..=2 {
            let mut spectrum = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_fft(source, -1, normalization, &mut spectrum),
                OK
            );
            let scale = match normalization {
                1 => 1.0 / 30.0f64.sqrt(),
                2 => 1.0 / 30.0,
                _ => 1.0,
            };
            let expected = reference
                .iter()
                .map(|value| *value * scale)
                .collect::<Vec<_>>();
            assert_complex_close(&complex_values(spectrum), &expected);

            let mut restored = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_ifft(spectrum, -1, normalization, &mut restored),
                OK
            );
            assert_complex_close(&complex_values(restored), &values);
            dispose(&[restored, spectrum]);
        }
        let empty = real_array(&[], &[0]);
        for normalization in 0..=2 {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_ifft(empty, -1, normalization, &mut output),
                DOMAIN_ERROR
            );
            assert!(output.is_null());
        }
        dispose(&[empty]);
        dispose(&[source]);
    }
}

#[test]
fn numpy_fft_identity_covers_every_upstream_length() {
    unsafe {
        for length in 1..512usize {
            let real_values = (0..length)
                .map(|index| (index as f64 * 0.113).sin())
                .collect::<Vec<_>>();
            let complex_values = real_values
                .iter()
                .enumerate()
                .map(|(index, value)| Complex64::new(*value, (index as f64 * 0.071).cos()))
                .collect::<Vec<_>>();
            let real = real_array(&real_values, &[length as i64]);
            let complex = complex_array(&complex_values, &[length as i64], 13);

            let mut spectrum = std::ptr::null_mut();
            let mut restored = std::ptr::null_mut();
            assert_eq!(terlan_ndarray_v1_fft(complex, -1, 0, &mut spectrum), OK);
            assert_eq!(terlan_ndarray_v1_ifft(spectrum, -1, 0, &mut restored), OK);
            assert_complex_close(&self::complex_values(restored), &complex_values);

            let mut real_spectrum = std::ptr::null_mut();
            let mut real_restored = std::ptr::null_mut();
            assert_eq!(terlan_ndarray_v1_rfft(real, -1, 0, &mut real_spectrum), OK);
            assert_eq!(
                terlan_ndarray_v1_irfft(real_spectrum, length as i64, -1, 0, &mut real_restored,),
                OK
            );
            for (actual, expected) in self::real_values(real_restored)
                .into_iter()
                .zip(&real_values)
            {
                close(Complex64::new(actual, 0.0), Complex64::new(*expected, 0.0));
            }
            dispose(&[
                real_restored,
                real_spectrum,
                restored,
                spectrum,
                complex,
                real,
            ]);
        }
    }
}

#[test]
fn numpy_sized_transforms_zero_pad_truncate_and_preserve_norms() {
    unsafe {
        let values = (0..30)
            .map(|index| (index as f64 * 0.127).sin())
            .collect::<Vec<_>>();
        let source = real_array(&values, &[30]);
        for length in [15usize, 30, 60] {
            let expected = (0..length)
                .map(|index| values.get(index).copied().unwrap_or(0.0))
                .collect::<Vec<_>>();
            for normalization in 0..=2 {
                let mut spectrum = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_fft_sized(
                        source,
                        length as i64,
                        -1,
                        normalization,
                        &mut spectrum,
                    ),
                    OK
                );
                let mut restored = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_ifft_sized(
                        spectrum,
                        length as i64,
                        -1,
                        normalization,
                        &mut restored,
                    ),
                    OK
                );
                assert_complex_close(
                    &self::complex_values(restored),
                    &expected
                        .iter()
                        .map(|value| Complex64::new(*value, 0.0))
                        .collect::<Vec<_>>(),
                );

                let mut real_spectrum = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_rfft_sized(
                        source,
                        length as i64,
                        -1,
                        normalization,
                        &mut real_spectrum,
                    ),
                    OK
                );
                let mut real_restored = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_irfft(
                        real_spectrum,
                        length as i64,
                        -1,
                        normalization,
                        &mut real_restored,
                    ),
                    OK
                );
                for (actual, expected) in
                    self::real_values(real_restored).into_iter().zip(&expected)
                {
                    close(Complex64::new(actual, 0.0), Complex64::new(*expected, 0.0));
                }

                let mut hermitian = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_ihfft_sized(
                        source,
                        length as i64,
                        -1,
                        normalization,
                        &mut hermitian,
                    ),
                    OK
                );
                let mut hermitian_restored = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_hfft(
                        hermitian,
                        length as i64,
                        -1,
                        normalization,
                        &mut hermitian_restored,
                    ),
                    OK
                );
                for (actual, expected) in self::real_values(hermitian_restored)
                    .into_iter()
                    .zip(&expected)
                {
                    close(Complex64::new(actual, 0.0), Complex64::new(*expected, 0.0));
                }
                dispose(&[
                    hermitian_restored,
                    hermitian,
                    real_restored,
                    real_spectrum,
                    restored,
                    spectrum,
                ]);
            }
        }
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_fft_sized(source, 0, -1, 0, &mut output),
            DOMAIN_ERROR
        );
        assert!(output.is_null());
        dispose(&[source]);
    }
}

#[test]
fn numpy_fft_multidimensional_axes_and_norms_round_trip() {
    unsafe {
        let shape = [2, 3, 4];
        let values = (0..24)
            .map(|index| (index as f64 * 0.23).sin() + index as f64 / 11.0)
            .collect::<Vec<_>>();
        let source = real_array(&values, &shape);
        let axes = [
            [0i64, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        for selected_axes in axes {
            for normalization in 0..=2 {
                let mut spectrum = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_fftn(
                        source,
                        selected_axes.as_ptr(),
                        3,
                        normalization,
                        &mut spectrum,
                    ),
                    OK
                );
                let mut restored = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_ifftn(
                        spectrum,
                        selected_axes.as_ptr(),
                        3,
                        normalization,
                        &mut restored,
                    ),
                    OK
                );
                for (actual, expected) in complex_values(restored).into_iter().zip(&values) {
                    close(actual, Complex64::new(*expected, 0.0));
                }
                dispose(&[restored, spectrum]);

                let real_axis = selected_axes[2] as usize;
                let mut real_spectrum = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_rfftn(
                        source,
                        selected_axes.as_ptr(),
                        3,
                        normalization,
                        &mut real_spectrum,
                    ),
                    OK
                );
                let mut real_restored = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_irfftn(
                        real_spectrum,
                        shape[real_axis],
                        selected_axes.as_ptr(),
                        3,
                        normalization,
                        &mut real_restored,
                    ),
                    OK
                );
                for (actual, expected) in real_values(real_restored).into_iter().zip(&values) {
                    close(Complex64::new(actual, 0.0), Complex64::new(*expected, 0.0));
                }
                dispose(&[real_restored, real_spectrum]);
            }
        }
        dispose(&[source]);
    }
}

#[test]
fn numpy_real_and_hermitian_norms_round_trip() {
    unsafe {
        let values = (0..30)
            .map(|index| (index as f64 * 0.31).cos())
            .collect::<Vec<_>>();
        let source = real_array(&values, &[30]);
        for normalization in 0..=2 {
            let mut spectrum = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_rfft(source, -1, normalization, &mut spectrum),
                OK
            );
            let mut restored = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_irfft(spectrum, 30, -1, normalization, &mut restored),
                OK
            );
            for (actual, expected) in real_values(restored).into_iter().zip(&values) {
                close(Complex64::new(actual, 0.0), Complex64::new(*expected, 0.0));
            }

            let mut compact = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_ihfft(source, -1, normalization, &mut compact),
                OK
            );
            let mut hermitian_restored = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_hfft(compact, 30, -1, normalization, &mut hermitian_restored,),
                OK
            );
            for (actual, expected) in real_values(hermitian_restored).into_iter().zip(&values) {
                close(Complex64::new(actual, 0.0), Complex64::new(*expected, 0.0));
            }
            dispose(&[hermitian_restored, compact, restored, spectrum]);
        }
        dispose(&[source]);
    }
}

#[test]
fn numpy_frequency_and_shift_helpers_cover_odd_even_axes_and_empty_dimensions() {
    unsafe {
        for length in [9i64, 10] {
            let mut frequencies = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_fftfreq(length, std::f64::consts::PI, &mut frequencies),
                OK
            );
            let expected = (0..length)
                .map(|index| {
                    let bin = if index <= (length - 1) / 2 {
                        index
                    } else {
                        index - length
                    };
                    bin as f64 / (length as f64 * std::f64::consts::PI)
                })
                .collect::<Vec<_>>();
            assert_eq!(real_values(frequencies), expected);

            let mut real_frequencies = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_rfftfreq(length, std::f64::consts::PI, &mut real_frequencies,),
                OK
            );
            assert_eq!(
                real_values(real_frequencies),
                (0..=length / 2)
                    .map(|index| index as f64 / (length as f64 * std::f64::consts::PI))
                    .collect::<Vec<_>>()
            );
            dispose(&[real_frequencies, frequencies]);
        }

        for rows in 0..16i64 {
            for columns in 0..16i64 {
                let values = (0..rows * columns)
                    .map(|value| value as f64)
                    .collect::<Vec<_>>();
                let source = real_array(&values, &[rows, columns]);
                let mut shifted = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_fftshift(source, std::ptr::null(), 0, &mut shifted),
                    OK
                );
                let mut restored = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_ifftshift(shifted, std::ptr::null(), 0, &mut restored,),
                    OK
                );
                assert_eq!(real_values(restored), values);
                dispose(&[restored, shifted, source]);
            }
        }
    }
}

#[test]
fn numpy_irfftn_supports_a_real_axis_that_is_not_last() {
    unsafe {
        let shape = [2, 8, 4, 3];
        let values = (0..192)
            .map(|index| (index as f64 * 0.07).sin())
            .collect::<Vec<_>>();
        let source = real_array(&values, &shape);
        let axes = [-2i64];
        let mut spectrum = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_rfftn(source, axes.as_ptr(), 1, 0, &mut spectrum),
            OK
        );
        let mut restored = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_irfftn(spectrum, 4, axes.as_ptr(), 1, 0, &mut restored),
            OK
        );
        for (actual, expected) in real_values(restored).into_iter().zip(values) {
            close(Complex64::new(actual, 0.0), Complex64::new(expected, 0.0));
        }
        dispose(&[restored, spectrum, source]);
    }
}

#[test]
fn numpy_fft_calls_are_thread_safe_without_shared_planner_state() {
    let workers = (0..16)
        .map(|worker| {
            std::thread::spawn(move || unsafe {
                let values = (0..128)
                    .map(|index| (index as f64 * 0.03 + worker as f64).sin())
                    .collect::<Vec<_>>();
                let source = real_array(&values, &[8, 16]);
                for operation in 0..4 {
                    let mut transformed = std::ptr::null_mut();
                    let status = match operation {
                        0 => terlan_ndarray_v1_fft(source, -1, 0, &mut transformed),
                        1 => terlan_ndarray_v1_ifft(source, -1, 0, &mut transformed),
                        2 => terlan_ndarray_v1_rfft(source, -1, 0, &mut transformed),
                        _ => {
                            let mut compact = std::ptr::null_mut();
                            assert_eq!(terlan_ndarray_v1_rfft(source, -1, 0, &mut compact), OK);
                            let status =
                                terlan_ndarray_v1_irfft(compact, 16, -1, 0, &mut transformed);
                            dispose(&[compact]);
                            status
                        }
                    };
                    assert_eq!(status, OK);
                    dispose(&[transformed]);
                }
                dispose(&[source]);
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().expect("FFT worker must not panic");
    }
}

#[test]
fn numpy_fft_layouts_match_for_all_admitted_precision_families() {
    unsafe {
        let shape = [2, 3, 4];
        let transposed_shape = [3, 2, 4];
        let real_values = (0..24)
            .map(|index| (index as f64 * 0.17).sin())
            .collect::<Vec<_>>();
        for input_dtype in [11, 3] {
            let source = real_array_as(&real_values, &shape, input_dtype);
            let mut strided = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_transpose(source, 0, 1, &mut strided),
                OK
            );
            let contiguous =
                real_array_as(&self::real_values(strided), &transposed_shape, input_dtype);
            assert_complex_transforms_equal(contiguous, strided);
            dispose(&[contiguous, strided, source]);
        }

        let complex_values = (0..24)
            .map(|index| Complex64::new((index as f64 * 0.11).cos(), (index as f64 * 0.07).sin()))
            .collect::<Vec<_>>();
        for input_dtype in [12, 13] {
            let source = complex_array(&complex_values, &shape, input_dtype);
            let mut strided = std::ptr::null_mut();
            assert_eq!(
                crate::ffi::terlan_ndarray_v1_transpose(source, 0, 1, &mut strided),
                OK
            );
            let contiguous = complex_array(
                &self::complex_values(strided),
                &transposed_shape,
                input_dtype,
            );
            assert_complex_transforms_equal(contiguous, strided);
            dispose(&[contiguous, strided, source]);
        }
    }
}
