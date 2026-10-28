use super::super::native_support::OK;
use super::*;

/// Creates one owned Float64 test array.
unsafe fn array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { owned_real(values, dimensions, 3).unwrap() }
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

/// Reads one three-array histogram result.
unsafe fn result(identities: *mut u64, length: usize) -> (Array, Array, Array) {
    assert_eq!(length, 3);
    let values = unsafe { arrays(identities, length) };
    (values[0], values[1], values[2])
}

/// Asserts equal-length floating vectors within one absolute tolerance.
fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn histogram2d_matches_numpy_counts_ranges_edges_density_empty_and_mixed_bins() {
    unsafe {
        let x = array(
            &[
                0.417022,
                0.72032449,
                0.00011437481,
                0.302332573,
                0.146755891,
            ],
            &[5],
        );
        let y = array(
            &[0.09233859, 0.18626021, 0.34556073, 0.39676747, 0.53881673],
            &[5],
        );
        let edges = array(
            &(0..=9).map(|index| index as f64 / 9.0).collect::<Vec<_>>(),
            &[10],
        );
        let mut output = std::ptr::null_mut();
        let mut length = 0;
        assert_eq!(
            terlan_ndarray_v1_histogram2d_edges(
                x,
                y,
                edges,
                edges,
                false,
                &mut output,
                &mut length
            ),
            OK
        );
        let (histogram, x_edges, y_edges) = result(output, length);
        assert_eq!(
            super::super::native_support::shape(histogram).unwrap(),
            [9, 9]
        );
        let mut expected = vec![0.0; 81];
        for (row, column) in [(3, 0), (6, 1), (0, 3), (2, 3), (1, 4)] {
            expected[row * 9 + column] = 1.0;
        }
        assert_eq!(floats(histogram, false).unwrap(), expected);
        close(
            &floats(x_edges, false).unwrap(),
            &floats(edges, false).unwrap(),
            0.0,
        );
        close(
            &floats(y_edges, false).unwrap(),
            &floats(edges, false).unwrap(),
            0.0,
        );
        delete_arrays(&[histogram, x_edges, y_edges]);

        let diagonal = array(
            &(0..10).map(|value| value as f64).collect::<Vec<_>>(),
            &[10],
        );
        assert_eq!(
            terlan_ndarray_v1_histogram2d(diagonal, diagonal, 10, false, &mut output, &mut length),
            OK
        );
        let (histogram, x_edges, y_edges) = result(output, length);
        let mut expected = vec![0.0; 100];
        for index in 0..10 {
            expected[index * 10 + index] = 1.0;
        }
        assert_eq!(floats(histogram, false).unwrap(), expected);
        close(
            &floats(x_edges, false).unwrap(),
            &(0..=10).map(|index| index as f64 * 0.9).collect::<Vec<_>>(),
            1.0e-12,
        );
        close(
            &floats(y_edges, false).unwrap(),
            &floats(x_edges, false).unwrap(),
            0.0,
        );
        delete_arrays(&[histogram, x_edges, y_edges]);

        let asym_x = array(&[1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0], &[8]);
        let asym_y = array(&[1.0, 3.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0], &[8]);
        let minima = array(&[0.0, 0.0], &[2]);
        let maxima = array(&[6.0, 5.0], &[2]);
        assert_eq!(
            terlan_ndarray_v1_histogram2d_range(
                asym_x,
                asym_y,
                6,
                5,
                minima,
                maxima,
                true,
                &mut output,
                &mut length
            ),
            OK
        );
        let (histogram, x_edges, y_edges) = result(output, length);
        close(
            &floats(histogram, false).unwrap(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.125, 0.0, 0.125, 0.0, 0.0, 0.0, 0.125, 0.0, 0.0,
                0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.125, 0.125, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.125,
            ],
            1.0e-12,
        );
        assert_eq!(
            floats(x_edges, false).unwrap(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            floats(y_edges, false).unwrap(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
        delete_arrays(&[histogram, x_edges, y_edges]);

        let density_x = array(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[9]);
        let density_y = array(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], &[9]);
        let variable_edges = array(&[1.0, 2.0, 3.0, 5.0], &[4]);
        assert_eq!(
            terlan_ndarray_v1_histogram2d_edges(
                density_x,
                density_y,
                variable_edges,
                variable_edges,
                true,
                &mut output,
                &mut length
            ),
            OK
        );
        let (histogram, x_edges, y_edges) = result(output, length);
        close(
            &floats(histogram, false).unwrap(),
            &[
                1.0 / 9.0,
                1.0 / 9.0,
                0.5 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                0.5 / 9.0,
                0.5 / 9.0,
                0.5 / 9.0,
                0.25 / 9.0,
            ],
            1.0e-12,
        );
        delete_arrays(&[histogram, x_edges, y_edges]);

        let outliers = array(&[1_000_001.0, 1_000_002.0], &[2]);
        let unit_minima = array(&[0.0, 0.0], &[2]);
        let unit_maxima = array(&[1.0, 1.0], &[2]);
        assert_eq!(
            terlan_ndarray_v1_histogram2d_range(
                outliers,
                outliers,
                4,
                5,
                unit_minima,
                unit_maxima,
                false,
                &mut output,
                &mut length
            ),
            OK
        );
        let (histogram, x_edges, y_edges) = result(output, length);
        assert_eq!(floats(histogram, false).unwrap(), vec![0.0; 20]);
        delete_arrays(&[histogram, x_edges, y_edges]);

        let empty = array(&[], &[0]);
        let unit_edges = array(&[0.0, 1.0], &[2]);
        assert_eq!(
            terlan_ndarray_v1_histogram2d_edges(
                empty,
                empty,
                unit_edges,
                unit_edges,
                false,
                &mut output,
                &mut length
            ),
            OK
        );
        let (histogram, x_edges, y_edges) = result(output, length);
        assert_eq!(floats(histogram, false).unwrap(), [0.0]);
        delete_arrays(&[histogram, x_edges, y_edges]);
        assert_eq!(
            terlan_ndarray_v1_histogram2d(empty, empty, 4, false, &mut output, &mut length),
            OK
        );
        let (histogram, x_edges, y_edges) = result(output, length);
        assert_eq!(floats(histogram, false).unwrap(), vec![0.0; 16]);
        assert_eq!(floats(x_edges, false).unwrap(), [0.0, 0.25, 0.5, 0.75, 1.0]);
        delete_arrays(&[histogram, x_edges, y_edges]);

        let mixed_x = array(
            &[
                0.0, 0.09207008, 0.64575234, 0.12875982, 0.47390599, 0.59944483, 1.0,
            ],
            &[7],
        );
        let mixed_y = array(
            &[
                0.0, 0.14344267, 0.48988575, 0.30558665, 0.44700682, 0.15886423, 1.0,
            ],
            &[7],
        );
        let tenths = array(
            &(0..=10)
                .map(|index| index as f64 / 10.0)
                .collect::<Vec<_>>(),
            &[11],
        );
        assert_eq!(
            terlan_ndarray_v1_histogram2d_x_edges(
                mixed_x,
                mixed_y,
                tenths,
                4,
                0.0,
                1.0,
                false,
                &mut output,
                &mut length
            ),
            OK
        );
        let (histogram, x_edges, y_edges) = result(output, length);
        assert_eq!(
            super::super::native_support::shape(histogram).unwrap(),
            [10, 4]
        );
        assert_eq!(floats(y_edges, false).unwrap(), [0.0, 0.25, 0.5, 0.75, 1.0]);
        delete_arrays(&[histogram, x_edges, y_edges]);
        assert_eq!(
            terlan_ndarray_v1_histogram2d_y_edges(
                mixed_x,
                mixed_y,
                4,
                0.0,
                1.0,
                tenths,
                false,
                &mut output,
                &mut length
            ),
            OK
        );
        let (histogram, x_edges, y_edges) = result(output, length);
        assert_eq!(
            super::super::native_support::shape(histogram).unwrap(),
            [4, 10]
        );
        assert_eq!(floats(x_edges, false).unwrap(), [0.0, 0.25, 0.5, 0.75, 1.0]);
        delete_arrays(&[histogram, x_edges, y_edges]);

        delete_arrays(&[
            x,
            y,
            edges,
            diagonal,
            asym_x,
            asym_y,
            minima,
            maxima,
            density_x,
            density_y,
            variable_edges,
            outliers,
            unit_minima,
            unit_maxima,
            empty,
            unit_edges,
            mixed_x,
            mixed_y,
            tenths,
        ]);
    }
}

#[test]
fn histogram2d_rejects_bad_counts_edges_ranges_shapes_and_outputs_without_leaks() {
    unsafe {
        let x = array(&[0.0, 1.0], &[2]);
        let short_y = array(&[0.0], &[1]);
        let invalid_edges = array(&[0.0, 1.0, 1.0], &[3]);
        let minima = array(&[0.0, 0.0], &[2]);
        let maxima = array(&[1.0, 1.0], &[2]);
        let mut output = 1usize as *mut u64;
        let mut length = 9usize;
        assert_eq!(
            terlan_ndarray_v1_histogram2d(x, short_y, 2, false, &mut output, &mut length),
            SHAPE_MISMATCH
        );
        assert!(output.is_null());
        assert_eq!(length, 0);
        assert_eq!(
            terlan_ndarray_v1_histogram2d(x, x, 0, false, &mut output, &mut length),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_histogram2d_edges(
                x,
                x,
                invalid_edges,
                invalid_edges,
                false,
                &mut output,
                &mut length
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_histogram2d_range(
                x,
                x,
                2,
                2,
                maxima,
                minima,
                false,
                &mut output,
                &mut length
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_histogram2d(x, x, 2, false, std::ptr::null_mut(), &mut length),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_histogram2d(x, x, 2, false, &mut output, std::ptr::null_mut()),
            INVALID_ARGUMENT
        );
        delete_arrays(&[x, short_y, invalid_edges, minima, maxima]);
    }
}
