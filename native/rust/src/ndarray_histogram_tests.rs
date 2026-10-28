//! Tests for NumPy-compatible one-dimensional histograms.

use super::super::native_support::{
    delete_arrays, dtype, floats, owned_bool, owned_int, owned_int_as, owned_real, shape,
};
use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Reads and owns one generated two-array return value.
unsafe fn result(values: *mut u64, length: usize) -> (Array, Array) {
    assert_eq!(length, 2);
    assert!(!values.is_null());
    let handles = unsafe { std::slice::from_raw_parts(values, length) };
    let result = (handles[0] as Array, handles[1] as Array);
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(values) };
    result
}

/// Builds one Float64 vector.
unsafe fn real(values: &[f64]) -> Array {
    unsafe { owned_real(values, &[values.len() as i64], 3).unwrap() }
}

/// Builds one Int64 vector.
unsafe fn integer(values: &[i64]) -> Array {
    unsafe { owned_int(values, &[values.len() as i64]).unwrap() }
}

/// Asserts approximate floating-point equality.
fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "{actual} != {expected}"
        );
    }
}

/// Builds an inclusive linear sequence with NumPy-compatible endpoint spacing.
fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 1 {
        return vec![start];
    }
    (0..count)
        .map(|index| start + (stop - start) * index as f64 / (count - 1) as f64)
        .collect()
}

/// Returns the selected number of bins for one estimator fixture.
unsafe fn selected_bins(values: &[f64], estimator: i64, range: Option<(f64, f64)>) -> usize {
    let array = unsafe { real(values) };
    let edges = unsafe { estimated_edges(array, estimator, range).unwrap() };
    unsafe { delete_arrays(&[array]) };
    edges.len() - 1
}

#[test]
fn inferred_edges_ranges_density_weights_and_right_edge_match_numpy() {
    unsafe {
        let values = real(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let mut output = std::ptr::null_mut();
        let mut length = 0;
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram(values, 5, false, &mut output, &mut length),
            0
        );
        let (counts, edges) = result(output, length);
        assert_eq!(floats(counts, false).unwrap(), [2.0; 5]);
        close(
            &floats(edges, false).unwrap(),
            &[0.0, 1.8, 3.6, 5.4, 7.2, 9.0],
        );
        assert_eq!(shape(counts).unwrap(), [5]);
        delete_arrays(&[counts, edges]);

        let custom_edges = real(&[0.0, 1.0, 3.0, 6.0, 10.0]);
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_edges(
                values,
                custom_edges,
                true,
                &mut output,
                &mut length,
            ),
            0
        );
        let (density, returned_edges) = result(output, length);
        close(&floats(density, false).unwrap(), &[0.1; 4]);
        assert_eq!(
            floats(returned_edges, false).unwrap(),
            [0.0, 1.0, 3.0, 6.0, 10.0]
        );
        delete_arrays(&[density, returned_edges]);

        let weighted_values = real(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let weights = real(&[2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_weighted_edges(
                weighted_values,
                weights,
                custom_edges,
                true,
                &mut output,
                &mut length,
            ),
            0
        );
        let (weighted, returned_edges) = result(output, length);
        close(&floats(weighted, false).unwrap(), &[0.2, 0.1, 0.1, 0.075]);
        delete_arrays(&[weighted, returned_edges]);

        let infinite_edges = real(&[0.0, 1.0, 3.0, 6.0, f64::INFINITY]);
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_edges(
                values,
                infinite_edges,
                true,
                &mut output,
                &mut length,
            ),
            0
        );
        let (infinite_density, returned_edges) = result(output, length);
        close(
            &floats(infinite_density, false).unwrap(),
            &[0.1, 0.1, 0.1, 0.0],
        );
        delete_arrays(&[infinite_density, returned_edges, infinite_edges]);

        let one_bin_values = integer(&[1, 2, 3, 4]);
        let one_bin_edges = integer(&[1, 2]);
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_edges(
                one_bin_values,
                one_bin_edges,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let (one_bin, returned_edges) = result(output, length);
        assert_eq!(floats(one_bin, false).unwrap(), [2.0]);
        delete_arrays(&[one_bin, returned_edges, one_bin_edges, one_bin_values]);

        let edge_case_values = integer(&[337, 404, 739, 806, 1007, 1811, 2012]);
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_range(
                edge_case_values,
                8296,
                2.0,
                2280.0,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let (edge_counts, edge_locations) = result(output, length);
        let counts = floats(edge_counts, false).unwrap();
        let edges = floats(edge_locations, false).unwrap();
        assert_eq!(counts.iter().sum::<f64>(), 7.0);
        for value in [337.0, 404.0, 739.0, 806.0, 1007.0, 1811.0, 2012.0] {
            let occupied = counts.iter().enumerate().find(|(index, count)| {
                **count > 0.0 && value >= edges[*index] && value < edges[*index + 1]
            });
            assert!(occupied.is_some(), "no occupied bin contains {value}");
        }
        delete_arrays(&[edge_counts, edge_locations, edge_case_values]);

        let type_values = real(&[1.0, 2.0, 2.0, 4.0]);
        let integer_weights = integer(&[4, 3, 2, 1]);
        let real_weights = real(&[4.0, 3.0, 2.0, 1.0]);
        for (weights, expected_dtype, density) in [
            (integer_weights, 2, false),
            (real_weights, 3, false),
            (integer_weights, 3, true),
        ] {
            assert_eq!(
                terlan_ndarray_v1_numpy_histogram_weighted_range(
                    type_values,
                    weights,
                    4,
                    1.0,
                    4.0,
                    density,
                    &mut output,
                    &mut length,
                ),
                0
            );
            let (typed_counts, typed_edges) = result(output, length);
            assert_eq!(dtype(typed_counts).unwrap(), expected_dtype);
            delete_arrays(&[typed_counts, typed_edges]);
        }
        delete_arrays(&[real_weights, integer_weights, type_values]);
        delete_arrays(&[weights, weighted_values, custom_edges, values]);
    }
}

#[test]
fn estimators_empty_constant_small_and_range_limited_data_match_numpy() {
    unsafe {
        let empty = real(&[]);
        let constant = real(&[1.0; 100]);
        let small = real(&[0.0, 1.0, 2.0]);
        let ranged = real(&[-100.0, 0.0, 1.0, 2.0, 100.0]);
        let mut output = std::ptr::null_mut();
        let mut length = 0;
        for estimator in 0..=7 {
            assert_eq!(
                terlan_ndarray_v1_numpy_histogram_estimator(
                    empty,
                    estimator,
                    false,
                    &mut output,
                    &mut length,
                ),
                0
            );
            let (counts, edges) = result(output, length);
            assert_eq!(floats(counts, false).unwrap(), [0.0]);
            assert_eq!(floats(edges, false).unwrap(), [0.0, 1.0]);
            delete_arrays(&[counts, edges]);

            assert_eq!(
                terlan_ndarray_v1_numpy_histogram_estimator(
                    constant,
                    estimator,
                    false,
                    &mut output,
                    &mut length,
                ),
                0
            );
            let (counts, edges) = result(output, length);
            assert_eq!(floats(counts, false).unwrap(), [100.0]);
            assert_eq!(floats(edges, false).unwrap(), [0.5, 1.5]);
            delete_arrays(&[counts, edges]);
        }
        let expected_small_bins = [2usize, 3, 3, 2, 1, 3, 2, 3];
        for (estimator, expected) in expected_small_bins.into_iter().enumerate() {
            assert_eq!(
                terlan_ndarray_v1_numpy_histogram_estimator(
                    small,
                    estimator as i64,
                    false,
                    &mut output,
                    &mut length,
                ),
                0
            );
            let (counts, edges) = result(output, length);
            assert_eq!(shape(counts).unwrap(), [expected as i64]);
            delete_arrays(&[counts, edges]);
        }
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_estimator_range(
                ranged,
                ESTIMATOR_AUTO,
                0.0,
                2.0,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let (counts, edges) = result(output, length);
        assert_eq!(floats(counts, false).unwrap().iter().sum::<f64>(), 3.0);
        assert_eq!(floats(edges, false).unwrap().first(), Some(&0.0));
        assert_eq!(floats(edges, false).unwrap().last(), Some(&2.0));
        delete_arrays(&[counts, edges]);
        delete_arrays(&[ranged, small, constant, empty]);
    }
}

#[test]
fn estimator_bin_counts_match_numpy_reference_fixtures() {
    unsafe {
        let estimators = [
            ESTIMATOR_FD,
            ESTIMATOR_SCOTT,
            ESTIMATOR_RICE,
            ESTIMATOR_STURGES,
            ESTIMATOR_DOANE,
            ESTIMATOR_SQRT,
            ESTIMATOR_AUTO,
            ESTIMATOR_STONE,
        ];
        let expected = [
            (50usize, [4, 4, 8, 7, 8, 8, 7, 2]),
            (500, [8, 8, 16, 10, 12, 23, 10, 9]),
            (5000, [17, 17, 35, 14, 17, 71, 17, 20]),
        ];
        for (count, expected) in expected {
            let mut values = linspace(-10.0, -1.0, count / 5 * 2);
            values.extend(linspace(1.0, 10.0, count / 5 * 3));
            for (&estimator, expected) in estimators.iter().zip(expected) {
                assert_eq!(selected_bins(&values, estimator, None), expected);
            }
        }

        let mut limited = vec![1.0; 1000];
        limited[..3].fill(0.0);
        limited[996..].fill(100.0);
        assert_eq!(selected_bins(&limited, ESTIMATOR_AUTO, None), 11);
        assert_eq!(selected_bins(&limited, ESTIMATOR_FD, None), 1);
        assert_eq!(selected_bins(&limited, ESTIMATOR_STURGES, None), 11);

        let mut outliers = linspace(-110.0, -100.0, 5);
        outliers.extend(linspace(-10.0, 10.0, 50));
        for (estimator, expected) in [
            (ESTIMATOR_FD, 21),
            (ESTIMATOR_SCOTT, 5),
            (ESTIMATOR_DOANE, 11),
            (ESTIMATOR_STONE, 6),
        ] {
            assert_eq!(selected_bins(&outliers, estimator, None), expected);
        }

        for count in [50usize, 500, 5000] {
            let mut values = linspace(-10.0, -1.0, count / 5 * 2);
            values.extend(linspace(1.0, 10.0, count / 5 * 3));
            values.extend(linspace(-100.0, -50.0, count));
            let expected = match count {
                50 => [8, 8, 15, 14, 14, 8],
                500 => [15, 16, 32, 20, 20, 80],
                _ => [33, 33, 69, 27, 33, 80],
            };
            for (&estimator, expected) in [
                ESTIMATOR_FD,
                ESTIMATOR_SCOTT,
                ESTIMATOR_RICE,
                ESTIMATOR_STURGES,
                ESTIMATOR_AUTO,
                ESTIMATOR_STONE,
            ]
            .iter()
            .zip(expected)
            {
                assert_eq!(
                    selected_bins(&values, estimator, Some((-20.0, 20.0))),
                    expected,
                    "range count={count} estimator={estimator}",
                );
            }
        }

        let signed = [-2.0, 0.0, 127.0];
        for estimator in 0..=7 {
            assert!(selected_bins(&signed, estimator, None) > 0);
        }
    }
}

#[test]
fn signed_extremes_edge_precision_and_estimator_convergence_match_numpy() {
    unsafe {
        let mut output = std::ptr::null_mut();
        let mut length = 0;
        for (dtype_id, lower, upper) in [
            (4, i8::MIN as i64 + 4, i8::MAX as i64 - 3),
            (5, i16::MIN as i64 + 4, i16::MAX as i64 - 3),
            (6, i32::MIN as i64 + 4, i32::MAX as i64 - 3),
            (2, i64::MIN + 4, i64::MAX - 3),
        ] {
            let values = owned_int_as(&[lower, upper], &[2], dtype_id).unwrap();
            assert_eq!(
                terlan_ndarray_v1_numpy_histogram(values, 2, false, &mut output, &mut length,),
                0
            );
            let (counts, edges) = result(output, length);
            assert_eq!(floats(counts, false).unwrap(), [1.0, 1.0]);
            assert_eq!(floats(edges, false).unwrap()[1], 0.0);
            assert_eq!(dtype(edges).unwrap(), 3);
            delete_arrays(&[counts, edges, values]);
        }

        for (dtype_id, epsilon) in [(14, f64::from(f32::EPSILON)), (11, f64::EPSILON)] {
            let values = owned_real(&[1.0], &[1], dtype_id).unwrap();
            for (lower, upper) in [(1.0 + epsilon, 2.0), (0.0, 1.0 - epsilon)] {
                assert_eq!(
                    terlan_ndarray_v1_numpy_histogram_range(
                        values,
                        1,
                        lower,
                        upper,
                        false,
                        &mut output,
                        &mut length,
                    ),
                    0
                );
                let (counts, edges) = result(output, length);
                assert_eq!(floats(counts, false).unwrap(), [1.0]);
                assert_eq!(dtype(edges).unwrap(), dtype_id);
                delete_arrays(&[counts, edges]);
            }
            delete_arrays(&[values]);
        }

        let normal = Normal::new(0.0, 2.0).unwrap();
        let sizes = [10usize, 22, 46, 100];
        let mut deviations = [0.0; 4];
        for seed in 0..10u64 {
            let mut generator = ChaCha8Rng::seed_from_u64(seed);
            for (index, size) in sizes.into_iter().enumerate() {
                let values = (0..size)
                    .map(|_| normal.sample(&mut generator))
                    .collect::<Vec<f64>>();
                let stone = selected_bins(&values, ESTIMATOR_STONE, None) as f64;
                let scott = selected_bins(&values, ESTIMATOR_SCOTT, None) as f64;
                deviations[index] += (stone / (stone + scott) - 0.5).abs() / 10.0;
            }
        }
        assert!(deviations[3] < deviations[0]);
        assert!(deviations[3] < 0.12, "deviations={deviations:?}");
    }
}

#[test]
fn float32_boolean_nan_dtype_and_immutability_regressions_match_numpy() {
    unsafe {
        let x = owned_real(&[276.318359, -69.593948, 21.329449], &[3], 11).unwrap();
        let y = owned_real(&[5005.689453, 4481.327637, 6010.369629], &[3], 11).unwrap();
        let mut output = std::ptr::null_mut();
        let mut length = 0;
        assert_eq!(
            super::super::histogram2d_extension::terlan_ndarray_v1_histogram2d(
                x,
                y,
                100,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let handles = std::slice::from_raw_parts(output, length)
            .iter()
            .map(|&handle| handle as Array)
            .collect::<Vec<_>>();
        crate::ffi::terlan_ndarray_v1_delete_handle_values(output);
        assert_eq!(floats(handles[0], false).unwrap().iter().sum::<f64>(), 3.0);
        delete_arrays(&handles);

        let booleans = owned_bool(&[1, 1, 0], &[3]).unwrap();
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram(booleans, 10, false, &mut output, &mut length,),
            0
        );
        let (counts, edges) = result(output, length);
        assert_eq!(dtype(counts).unwrap(), 2);
        assert_eq!(floats(counts, false).unwrap().iter().sum::<f64>(), 3.0);
        delete_arrays(&[counts, edges]);

        let with_nan = real(&[0.0, 1.0, f64::NAN]);
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_estimator(
                with_nan,
                ESTIMATOR_AUTO,
                false,
                &mut output,
                &mut length,
            ),
            DOMAIN_ERROR
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_estimator_range(
                with_nan,
                ESTIMATOR_AUTO,
                0.0,
                1.0,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let (counts, edges) = result(output, length);
        assert_eq!(floats(counts, false).unwrap().iter().sum::<f64>(), 2.0);
        delete_arrays(&[counts, edges]);

        let all_nan = real(&[f64::NAN, f64::NAN]);
        let unit_edges = real(&[0.0, 1.0]);
        for (input, expected) in [(with_nan, 2.0), (all_nan, 0.0)] {
            assert_eq!(
                terlan_ndarray_v1_numpy_histogram_edges(
                    input,
                    unit_edges,
                    false,
                    &mut output,
                    &mut length,
                ),
                0
            );
            let (counts, returned_edges) = result(output, length);
            assert_eq!(floats(counts, false).unwrap().iter().sum::<f64>(), expected);
            delete_arrays(&[counts, returned_edges]);
        }

        let immutable = real(&[1.3, 2.5, 2.3]);
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_range(
                immutable,
                100,
                -10.0,
                10.0,
                false,
                &mut output,
                &mut length,
            ),
            0
        );
        let (counts, edges) = result(output, length);
        assert_eq!(floats(immutable, false).unwrap(), [1.3, 2.5, 2.3]);
        delete_arrays(&[
            counts, edges, immutable, unit_edges, all_nan, with_nan, booleans, y, x,
        ]);
    }
}

#[test]
fn histogram_rejects_bad_edges_ranges_weights_estimators_and_outputs() {
    unsafe {
        let values = integer(&[1, 2, 3]);
        let short_weights = real(&[1.0, 2.0]);
        let reversed = real(&[0.0, 2.0, 1.0]);
        let matrix_edges = owned_real(&[0.0, 1.0, 2.0, 3.0], &[2, 2], 3).unwrap();
        let mut output = std::ptr::null_mut();
        let mut length = 0;
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_range(
                values,
                0,
                0.0,
                1.0,
                false,
                &mut output,
                &mut length,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_range(
                values,
                2,
                f64::NAN,
                1.0,
                false,
                &mut output,
                &mut length,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_range(
                values,
                2,
                0.0,
                f64::INFINITY,
                false,
                &mut output,
                &mut length,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_range(
                values,
                2,
                1.0,
                1.0,
                false,
                &mut output,
                &mut length,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_edges(
                values,
                reversed,
                false,
                &mut output,
                &mut length,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_edges(
                values,
                matrix_edges,
                false,
                &mut output,
                &mut length,
            ),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_weighted(
                values,
                short_weights,
                2,
                false,
                &mut output,
                &mut length,
            ),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram_estimator(
                values,
                99,
                false,
                &mut output,
                &mut length,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_numpy_histogram(values, 2, false, std::ptr::null_mut(), &mut length),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        delete_arrays(&[matrix_edges, reversed, short_weights, values]);
    }
}
