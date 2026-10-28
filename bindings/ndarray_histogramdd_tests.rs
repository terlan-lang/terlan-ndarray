//! Tests for NumPy-compatible multidimensional histograms.

use super::super::native_support::{delete_arrays, dtype, floats, owned_int, owned_real, shape};
use super::*;

/// Reads and owns one generated transferred array list.
unsafe fn result(values: *mut u64, length: usize, expected: usize) -> Vec<Array> {
    assert_eq!(length, expected);
    assert!(!values.is_null());
    let arrays = unsafe { std::slice::from_raw_parts(values, length) }
        .iter()
        .map(|value| *value as Array)
        .collect::<Vec<_>>();
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(values) };
    arrays
}

/// Compares two Float64 slices under one strict numerical tolerance.
fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn histogramdd_uniform_ranges_shapes_weights_empty_and_constants_match_numpy() {
    unsafe {
        let samples = owned_real(
            &[
                -0.5, 0.5, 1.5, -0.5, 1.5, 2.5, -0.5, 2.5, 0.5, 0.5, 0.5, 1.5, 0.5, 1.5, 2.5, 0.5,
                2.5, 2.5,
            ],
            &[6, 3],
            3,
        )
        .unwrap();
        let bins = owned_int(&[2, 3, 3], &[3]).unwrap();
        let minima = owned_real(&[-1.0, 0.0, 0.0], &[3], 3).unwrap();
        let maxima = owned_real(&[1.0, 3.0, 3.0], &[3], 3).unwrap();
        let mut output = std::ptr::null_mut();
        let mut length = 0;
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_range_arrays(
                samples,
                bins,
                minima,
                maxima,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let arrays = result(output, length, 4);
        assert_eq!(shape(arrays[0]).unwrap(), [2, 3, 3]);
        assert_eq!(
            floats(arrays[0], false).unwrap(),
            [
                0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 1.0,
            ]
        );
        delete_arrays(&arrays);

        let shape_samples = owned_real(
            &(0..40).map(|value| value as f64 / 40.0).collect::<Vec<_>>(),
            &[10, 4],
            3,
        )
        .unwrap();
        for dimensions in [[7, 4, 5, 6], [4, 5, 7, 6], [6, 5, 4, 7]] {
            let bins = owned_int(&dimensions, &[4]).unwrap();
            assert_eq!(
                terlan_ndarray_v1_numpy_histogramdd_bins_array(
                    shape_samples,
                    bins,
                    false,
                    &mut output,
                    &mut length,
                ),
                0
            );
            let arrays = result(output, length, 5);
            assert_eq!(shape(arrays[0]).unwrap(), dimensions);
            delete_arrays(&arrays);
            delete_arrays(&[bins]);
        }

        let weighted_samples =
            owned_real(&[0.0, 0.0, 0.2, 0.2, 0.8, 0.8, 1.0, 1.0], &[4, 2], 3).unwrap();
        let ones = owned_real(&[1.0; 4], &[4], 3).unwrap();
        let twos = owned_int(&[2; 4], &[4]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd(
                weighted_samples,
                2,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let baseline = result(output, length, 3);
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_weighted(
                weighted_samples,
                ones,
                2,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let weighted = result(output, length, 3);
        assert_eq!(
            floats(weighted[0], false).unwrap(),
            floats(baseline[0], false).unwrap()
        );
        delete_arrays(&weighted);
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_weighted(
                weighted_samples,
                twos,
                2,
                true,
                &mut output,
                &mut length,
            ),
            0
        );
        let weighted_density = result(output, length, 3);
        assert_eq!(
            floats(weighted_density[0], false).unwrap(),
            floats(baseline[0], false).unwrap()
        );
        delete_arrays(&[baseline[1], baseline[2]]);
        delete_arrays(&[baseline[0]]);
        delete_arrays(&weighted_density);

        let empty = owned_real(&[], &[0, 3], 3).unwrap();
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd(empty, 2, false, &mut output, &mut length,),
            0
        );
        let empty_result = result(output, length, 4);
        assert_eq!(shape(empty_result[0]).unwrap(), [2, 2, 2]);
        assert_eq!(floats(empty_result[0], false).unwrap(), [0.0; 8]);
        for edges in &empty_result[1..] {
            assert_eq!(floats(*edges, false).unwrap(), [0.0, 0.5, 1.0]);
        }
        delete_arrays(&empty_result);

        let constant = owned_int(&[0; 20], &[10, 2]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd(constant, 2, false, &mut output, &mut length,),
            0
        );
        let constant_result = result(output, length, 3);
        assert_eq!(floats(constant_result[1], false).unwrap(), [-0.5, 0.0, 0.5]);
        assert_eq!(floats(constant_result[2], false).unwrap(), [-0.5, 0.0, 0.5]);
        delete_arrays(&constant_result);
        delete_arrays(&[
            constant,
            empty,
            twos,
            ones,
            weighted_samples,
            shape_samples,
            maxima,
            minima,
            bins,
            samples,
        ]);
    }
}

#[test]
fn histogramdd_explicit_edges_density_infinities_equal_edges_and_integers_match_numpy() {
    unsafe {
        let mut output = std::ptr::null_mut();
        let mut length = 0;
        let samples = owned_real(
            &[
                -0.5, 0.5, 1.5, -0.5, 1.5, 2.5, -0.5, 2.5, 0.5, 0.5, 0.5, 1.5, 0.5, 1.5, 2.5, 0.5,
                2.5, 2.5,
            ],
            &[6, 3],
            3,
        )
        .unwrap();
        let packed = owned_real(
            &[-2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
            &[11],
            3,
        )
        .unwrap();
        let offsets = owned_int(&[0, 3, 7, 11], &[4]).unwrap();
        let dtypes = owned_int(&[3, 3, 3], &[3]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_edges_packed(
                samples,
                packed,
                offsets,
                dtypes,
                true,
                &mut output,
                &mut length,
            ),
            0
        );
        let density = result(output, length, 4);
        close(
            &floats(density[0], false).unwrap(),
            &[
                0.0,
                1.0 / 12.0,
                0.0,
                0.0,
                0.0,
                1.0 / 12.0,
                1.0 / 12.0,
                0.0,
                0.0,
                0.0,
                1.0 / 12.0,
                0.0,
                0.0,
                0.0,
                1.0 / 12.0,
                0.0,
                0.0,
                1.0 / 12.0,
            ],
        );
        delete_arrays(&density);

        let inf_samples = owned_int(&[0, 1, 2, 3, 4, 5], &[3, 2]).unwrap();
        let inf_edges = owned_real(
            &[0.0, 4.0 / 3.0, 8.0 / 3.0, 4.0, f64::NEG_INFINITY, 2.0, 10.0],
            &[7],
            3,
        )
        .unwrap();
        let inf_offsets = owned_int(&[0, 4, 7], &[3]).unwrap();
        let inf_dtypes = owned_int(&[3, 3], &[2]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_edges_packed(
                inf_samples,
                inf_edges,
                inf_offsets,
                inf_dtypes,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let inf_result = result(output, length, 3);
        assert_eq!(
            floats(inf_result[0], false).unwrap(),
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
        );
        delete_arrays(&inf_result);

        let right_samples =
            owned_real(&[0.9999999995, 1.0, 1.0000000001, 1.0001], &[4, 1], 3).unwrap();
        let right_edges = owned_real(&[0.0, 0.5, 1.0], &[3], 3).unwrap();
        let right_offsets = owned_int(&[0, 3], &[2]).unwrap();
        let right_dtypes = owned_int(&[3], &[1]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_edges_packed(
                right_samples,
                right_edges,
                right_offsets,
                right_dtypes,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let right_result = result(output, length, 2);
        assert_eq!(floats(right_result[0], false).unwrap(), [0.0, 2.0]);
        delete_arrays(&right_result);

        let equal_samples = owned_int(&[0, 0, 1, 1, 2, 2], &[3, 2]).unwrap();
        let equal_edges = owned_int(&[0, 2, 2, 0, 2], &[5]).unwrap();
        let equal_offsets = owned_int(&[0, 3, 5], &[3]).unwrap();
        let equal_dtypes = owned_int(&[2, 2], &[2]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_edges_packed(
                equal_samples,
                equal_edges,
                equal_offsets,
                equal_dtypes,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let equal_result = result(output, length, 3);
        assert_eq!(floats(equal_result[0], false).unwrap(), [2.0, 1.0]);
        assert_eq!(dtype(equal_result[1]).unwrap(), 2);
        delete_arrays(&equal_result);

        let big = 1_i64 << 60;
        let large_samples = owned_int(&[0, big], &[1, 2]).unwrap();
        let large_edges = owned_int(&[-1, 1, big - 1, big + 1], &[4]).unwrap();
        let large_offsets = owned_int(&[0, 2, 4], &[3]).unwrap();
        let large_dtypes = owned_int(&[2, 2], &[2]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_edges_packed(
                large_samples,
                large_edges,
                large_offsets,
                large_dtypes,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let large_result = result(output, length, 3);
        assert_eq!(floats(large_result[0], false).unwrap(), [1.0]);
        assert_eq!(dtype(large_result[1]).unwrap(), 2);
        assert_eq!(dtype(large_result[2]).unwrap(), 2);
        delete_arrays(&large_result);

        let area_samples = owned_int(
            &[
                7, 1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7, 1, 7,
                1, 7, 1, 7,
            ],
            &[16, 2],
        )
        .unwrap();
        let area_edges = owned_int(&[0, 6, 8, 0, 2, 8], &[6]).unwrap();
        let area_offsets = owned_int(&[0, 3, 6], &[3]).unwrap();
        let area_dtypes = owned_int(&[2, 2], &[2]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_edges_packed(
                area_samples,
                area_edges,
                area_offsets,
                area_dtypes,
                true,
                &mut output,
                &mut length,
            ),
            0
        );
        let area_result = result(output, length, 3);
        close(&floats(area_result[0], false).unwrap(), &[1.0 / 64.0; 4]);
        delete_arrays(&area_result);

        delete_arrays(&[
            area_dtypes,
            area_offsets,
            area_edges,
            area_samples,
            large_dtypes,
            large_offsets,
            large_edges,
            large_samples,
            equal_dtypes,
            equal_offsets,
            equal_edges,
            equal_samples,
            right_dtypes,
            right_offsets,
            right_edges,
            right_samples,
            inf_dtypes,
            inf_offsets,
            inf_edges,
            inf_samples,
            dtypes,
            offsets,
            packed,
            samples,
        ]);
    }
}

#[test]
fn histogramdd_rejects_invalid_shapes_bins_ranges_edges_weights_and_outputs() {
    unsafe {
        let samples = owned_real(&[0.0; 8], &[2, 4], 3).unwrap();
        let short_bins = owned_int(&[1, 1, 1], &[3]).unwrap();
        let bad_bins = owned_int(&[1, 0, 1, 1], &[4]).unwrap();
        let minima = owned_real(&[0.0; 4], &[4], 3).unwrap();
        let maxima = owned_real(&[1.0, 1.0, 1.0, f64::INFINITY], &[4], 3).unwrap();
        let packed = owned_real(&[0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0], &[8], 3).unwrap();
        let offsets = owned_int(&[0, 2, 5, 6, 8], &[5]).unwrap();
        let dtypes = owned_int(&[3; 4], &[4]).unwrap();
        let weights = owned_real(&[1.0], &[1], 3).unwrap();
        let mut output = std::ptr::null_mut();
        let mut length = 0;
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_bins_array(
                samples,
                short_bins,
                false,
                &mut output,
                &mut length,
            ),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_bins_array(
                samples,
                bad_bins,
                false,
                &mut output,
                &mut length,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_range_arrays(
                samples,
                bad_bins,
                minima,
                maxima,
                false,
                &mut output,
                &mut length,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_edges_packed(
                samples,
                packed,
                offsets,
                dtypes,
                false,
                &mut output,
                &mut length,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd_weighted(
                samples,
                weights,
                2,
                false,
                &mut output,
                &mut length,
            ),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogramdd(
                samples,
                2,
                false,
                std::ptr::null_mut(),
                &mut length,
            ),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        delete_arrays(&[
            weights, dtypes, offsets, packed, maxima, minima, bad_bins, short_bins, samples,
        ]);
    }
}
