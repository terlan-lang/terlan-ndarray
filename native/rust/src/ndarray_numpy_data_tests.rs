use super::*;

unsafe fn array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, dimensions, 3).unwrap() }
}

unsafe fn values(array: Array) -> Vec<f64> {
    unsafe { super::super::native_support::floats(array, false).unwrap() }
}

unsafe fn packable(values: &[i64], dimensions: &[i64], dtype: i32) -> Array {
    if dtype == 1 {
        let values = values
            .iter()
            .map(|value| u8::from(*value != 0))
            .collect::<Vec<_>>();
        unsafe { super::super::native_support::owned_bool(&values, dimensions).unwrap() }
    } else {
        unsafe { super::super::native_support::owned_int_as(values, dimensions, dtype).unwrap() }
    }
}

unsafe fn assert_array(array: Array, dimensions: &[i64], expected: &[i64]) {
    assert_eq!(
        unsafe { super::super::native_support::shape(array) }.unwrap(),
        dimensions
    );
    assert_eq!(
        unsafe { super::super::native_support::dtype(array) }.unwrap(),
        7
    );
    assert_eq!(
        unsafe { super::super::native_support::ints(array) }.unwrap(),
        expected
    );
}

fn flat_pack_reference(values: &[i64], little: bool) -> Vec<i64> {
    values
        .chunks(8)
        .map(|chunk| {
            chunk.iter().enumerate().fold(0, |byte, (index, value)| {
                byte | if *value == 0 {
                    0
                } else {
                    1 << if little { index } else { 7 - index }
                }
            })
        })
        .collect()
}

fn axis_prefix(values: &[i64], axis: i64, length: usize) -> Vec<i64> {
    let mut output = Vec::with_capacity(if axis == 0 { length * 7 } else { 7 * length });
    if axis == 0 {
        for row in 0..length {
            for column in 0..7 {
                output.push(if row < 7 { values[row * 7 + column] } else { 0 });
            }
        }
    } else {
        for row in 0..7 {
            for column in 0..length {
                output.push(if column < 7 {
                    values[row * 7 + column]
                } else {
                    0
                });
            }
        }
    }
    output
}

fn axis_all(values: &[i64], axis: i64, little: bool) -> Vec<i64> {
    if !little {
        return axis_prefix(values, axis, 8);
    }
    let mut output = Vec::with_capacity(56);
    if axis == 0 {
        output.resize(7, 0);
        for row in (0..7).rev() {
            output.extend_from_slice(&values[row * 7..row * 7 + 7]);
        }
    } else {
        for row in 0..7 {
            output.push(0);
            output.extend(values[row * 7..row * 7 + 7].iter().rev());
        }
    }
    output
}

#[test]
fn nan_reductions_cumulative_operations_and_indexes_are_deterministic() {
    unsafe {
        let source = array(&[3.0, f64::NAN, 1.0, 2.0], &[4]);
        let mut maximum = -1;
        let mut minimum = -1;
        let mut cumulative = std::ptr::null_mut();
        let mut median = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_nanargmax(source, &mut maximum), 0);
        assert_eq!(terlan_ndarray_v1_nanargmin(source, &mut minimum), 0);
        assert_eq!((maximum, minimum), (0, 2));
        assert_eq!(terlan_ndarray_v1_nancumsum(source, 0, &mut cumulative), 0);
        assert_eq!(values(cumulative), [3.0, 3.0, 4.0, 6.0]);
        assert_eq!(terlan_ndarray_v1_nanmedian(source, &mut median), 0);
        assert_eq!(values(median), [2.0]);
        super::super::native_support::delete_arrays(&[source, cumulative, median]);
    }
}

#[test]
fn nan_integer_types_mutation_scalar_percentile_and_quantile_match_numpy() {
    unsafe {
        let input = [127, 39, 93, 87, 46];
        let sums = [127, 166, 259, 346, 392];
        let products = [127, 4_953, 460_629, 40_074_723, 1_843_437_258];
        for dtype in [4, 5, 6, 2, 7, 8, 9, 10] {
            let source = packable(&input, &[5], dtype);
            let mut minimum = -1;
            let mut maximum = -1;
            let mut cumulative_sum = std::ptr::null_mut();
            let mut cumulative_product = std::ptr::null_mut();
            assert_eq!(terlan_ndarray_v1_nanargmin(source, &mut minimum), 0);
            assert_eq!(terlan_ndarray_v1_nanargmax(source, &mut maximum), 0);
            assert_eq!((minimum, maximum), (1, 0));
            assert_eq!(
                terlan_ndarray_v1_nancumsum(source, 0, &mut cumulative_sum),
                0
            );
            assert_eq!(
                terlan_ndarray_v1_nancumprod(source, 0, &mut cumulative_product),
                0
            );
            assert_eq!(
                super::super::native_support::ints(cumulative_sum).unwrap(),
                sums
            );
            assert_eq!(
                super::super::native_support::ints(cumulative_product).unwrap(),
                products
            );
            assert_eq!(super::super::native_support::ints(source).unwrap(), input);
            super::super::native_support::delete_arrays(&[
                source,
                cumulative_sum,
                cumulative_product,
            ]);
        }

        let source = array(&[f64::NAN, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], &[9]);
        let original = values(source);
        let mut median = std::ptr::null_mut();
        let mut percentile = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_nanmedian(source, &mut median), 0);
        assert_eq!(
            terlan_ndarray_v1_nanpercentile(source, 50.0, &mut percentile),
            0
        );
        assert!(
            values(source)
                .iter()
                .zip(&original)
                .all(|(actual, expected)| actual == expected
                    || (actual.is_nan() && expected.is_nan()))
        );
        assert_eq!(values(median), [1.75]);
        assert_eq!(values(percentile), [1.75]);

        for (quantile, expected) in [(0.0, 0.0), (1.0, 3.5), (0.5, 1.75)] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_nanquantile(source, quantile, &mut output),
                0
            );
            assert_eq!(values(output), [expected]);
            super::super::native_support::delete_arrays(&[output]);
        }
        let scalar = array(&[0.0], &[]);
        let mut scalar_percentile = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_nanpercentile(scalar, 100.0, &mut scalar_percentile),
            0
        );
        assert_eq!(values(scalar_percentile), [0.0]);
        let range = array(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[6]);
        let mut range_percentile = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_nanpercentile(range, 50.0, &mut range_percentile),
            0
        );
        assert_eq!(values(range_percentile), [2.5]);
        super::super::native_support::delete_arrays(&[
            source,
            median,
            percentile,
            scalar,
            scalar_percentile,
            range,
            range_percentile,
        ]);
    }
}

#[test]
fn byte_ingress_and_bit_packing_round_trip_exactly() {
    unsafe {
        let bytes = [0x80u8, 0x55];
        let mut source = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_frombuffer(bytes.as_ptr(), 2, &mut source),
            0
        );
        let mut unpacked = std::ptr::null_mut();
        let mut packed = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_unpackbits(source, 16, 0, &mut unpacked),
            0
        );
        assert_eq!(terlan_ndarray_v1_packbits(unpacked, 0, &mut packed), 0);
        assert_eq!(
            super::super::native_support::ints(packed).unwrap(),
            [128, 85]
        );
        super::super::native_support::delete_arrays(&[source, unpacked, packed]);
    }
}

#[test]
fn bit_packing_matches_numpy_axes_orders_empty_shapes_and_negative_counts() {
    unsafe {
        let input = [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1];
        for dtype in [1, 4, 7, 5, 8, 6, 9, 2, 10] {
            let source = packable(&input, &[2, 2, 3], dtype);
            let mut packed = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_packbits_axis(source, -1, 0, &mut packed),
                0
            );
            assert_eq!(
                super::super::native_support::shape(packed).unwrap(),
                [2, 2, 1]
            );
            assert_eq!(super::super::native_support::dtype(packed).unwrap(), 7);
            assert_eq!(
                super::super::native_support::ints(packed).unwrap(),
                [160, 64, 192, 32]
            );
            super::super::native_support::delete_arrays(&[source, packed]);
        }

        let bytes = super::super::native_support::owned_int_as(&[2, 7, 23], &[3, 1], 7).unwrap();
        let mut unpacked = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_unpackbits_axis_all(bytes, 1, 0, &mut unpacked),
            0
        );
        assert_eq!(
            super::super::native_support::shape(unpacked).unwrap(),
            [3, 8]
        );
        assert_eq!(super::super::native_support::dtype(unpacked).unwrap(), 7);
        assert_eq!(
            super::super::native_support::ints(unpacked).unwrap(),
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1,]
        );
        let mut little = std::ptr::null_mut();
        let mut little_roundtrip = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_unpackbits_axis_all(bytes, 1, 1, &mut little),
            0
        );
        assert_eq!(
            terlan_ndarray_v1_packbits_axis(little, 1, 1, &mut little_roundtrip),
            0
        );
        assert_eq!(
            super::super::native_support::ints(little_roundtrip).unwrap(),
            [2, 7, 23]
        );

        let empty_shapes = [
            vec![0],
            vec![10, 20, 0],
            vec![10, 0, 20],
            vec![0, 10, 20],
            vec![20, 0, 0],
            vec![0, 20, 0],
            vec![0, 0, 20],
            vec![0, 0, 0],
        ];
        for dtype in [1, 4, 7, 5, 8, 6, 9, 2, 10] {
            for dimensions in &empty_shapes {
                let empty = packable(&[], dimensions, dtype);
                let mut flat = std::ptr::null_mut();
                assert_eq!(terlan_ndarray_v1_packbits(empty, 0, &mut flat), 0);
                assert_eq!(super::super::native_support::shape(flat).unwrap(), [0]);
                for axis in 0..dimensions.len() {
                    let mut packed_empty = std::ptr::null_mut();
                    assert_eq!(
                        terlan_ndarray_v1_packbits_axis(empty, axis as i64, 0, &mut packed_empty,),
                        0
                    );
                    let mut expected = dimensions.clone();
                    expected[axis] = (expected[axis] + 7) / 8;
                    assert_eq!(
                        super::super::native_support::shape(packed_empty).unwrap(),
                        expected
                    );
                    super::super::native_support::delete_arrays(&[packed_empty]);
                }
                super::super::native_support::delete_arrays(&[flat, empty]);
            }
        }

        let empty_bytes = super::super::native_support::owned_int_as(&[], &[10, 0, 20], 7).unwrap();
        let mut unpacked_empty = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_unpackbits_axis_all(empty_bytes, 1, 0, &mut unpacked_empty),
            0
        );
        assert_eq!(
            super::super::native_support::shape(unpacked_empty).unwrap(),
            [10, 0, 20]
        );
        let flat_empty = super::super::native_support::owned_int_as(&[], &[0], 7).unwrap();
        let mut flat_unpacked_empty = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_unpackbits_all(flat_empty, 0, &mut flat_unpacked_empty),
            0
        );
        assert_array(flat_unpacked_empty, &[0], &[]);
        let unpack_empty_shapes = [
            (vec![0], vec![vec![0]]),
            (
                vec![16, 24, 0],
                vec![vec![2, 24, 0], vec![16, 3, 0], vec![16, 24, 0]],
            ),
            (
                vec![16, 0, 24],
                vec![vec![2, 0, 24], vec![16, 0, 24], vec![16, 0, 3]],
            ),
            (
                vec![0, 16, 24],
                vec![vec![0, 16, 24], vec![0, 2, 24], vec![0, 16, 3]],
            ),
            (
                vec![24, 0, 0],
                vec![vec![3, 0, 0], vec![24, 0, 0], vec![24, 0, 0]],
            ),
            (
                vec![0, 24, 0],
                vec![vec![0, 24, 0], vec![0, 3, 0], vec![0, 24, 0]],
            ),
            (
                vec![0, 0, 24],
                vec![vec![0, 0, 24], vec![0, 0, 24], vec![0, 0, 3]],
            ),
            (
                vec![0, 0, 0],
                vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
            ),
        ];
        for (output_shape, input_shapes) in unpack_empty_shapes {
            for (axis, input_shape) in input_shapes.iter().enumerate() {
                let empty =
                    super::super::native_support::owned_int_as(&[], input_shape, 7).unwrap();
                let mut output = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_unpackbits_axis_all(empty, axis as i64, 0, &mut output),
                    0
                );
                assert_array(output, &output_shape, &[]);
                super::super::native_support::delete_arrays(&[empty, output]);
            }
        }

        let seven_bytes = super::super::native_support::owned_int_as(
            &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x80],
            &[7],
            7,
        )
        .unwrap();
        let mut shortened = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_unpackbits_axis(seven_bytes, 0, -1, 0, &mut shortened),
            0
        );
        assert_eq!(
            super::super::native_support::shape(shortened).unwrap(),
            [55]
        );
        assert_eq!(
            super::super::native_support::ints(shortened).unwrap()[54],
            0
        );

        let mut rejected = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_unpackbits_axis(seven_bytes, 0, -57, 0, &mut rejected),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_unpackbits_axis_all(seven_bytes, 0, 10, &mut rejected),
            INVALID_ARGUMENT
        );
        let floating = array(&[1.0, 0.0], &[2]);
        assert_eq!(
            terlan_ndarray_v1_packbits(floating, 0, &mut rejected),
            DOMAIN_ERROR
        );
        assert_eq!(
            terlan_ndarray_v1_unpackbits_axis_all(floating, 0, 0, &mut rejected),
            DOMAIN_ERROR
        );
        assert!(rejected.is_null());
        super::super::native_support::delete_arrays(&[
            bytes,
            unpacked,
            little,
            little_roundtrip,
            empty_bytes,
            unpacked_empty,
            flat_empty,
            flat_unpacked_empty,
            seven_bytes,
            shortened,
            floating,
        ]);
    }
}

#[test]
fn bit_packing_handles_large_axes_strides_remainders_and_nonzero_values() {
    unsafe {
        let source_values = (0..900)
            .map(|index| i64::from((index * 37 + index / 7 + 11) % 5 != 0))
            .collect::<Vec<_>>();
        for bit_order in 0..=1 {
            for dtype in [1, 4, 7, 5, 8, 6, 9, 2, 10] {
                let source = packable(&source_values, &[900], dtype);
                let mut packed = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_packbits(source, bit_order, &mut packed),
                    0
                );
                assert_array(
                    packed,
                    &[113],
                    &flat_pack_reference(&source_values, bit_order == 1),
                );
                let mut unpacked = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_unpackbits(packed, 900, bit_order, &mut unpacked),
                    0
                );
                assert_array(unpacked, &[900], &source_values);
                for removed in 1..16 {
                    let prefix = &source_values[..source_values.len() - removed];
                    let prefix_array = packable(prefix, &[prefix.len() as i64], dtype);
                    let mut prefix_packed = std::ptr::null_mut();
                    assert_eq!(
                        terlan_ndarray_v1_packbits(prefix_array, bit_order, &mut prefix_packed),
                        0
                    );
                    assert_eq!(
                        super::super::native_support::ints(prefix_packed)
                            .unwrap()
                            .last(),
                        flat_pack_reference(prefix, bit_order == 1).last()
                    );
                    super::super::native_support::delete_arrays(&[prefix_array, prefix_packed]);
                }
                super::super::native_support::delete_arrays(&[source, packed, unpacked]);

                let nonzero = source_values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        if *value == 0 {
                            0
                        } else {
                            (index % 97 + 1) as i64
                        }
                    })
                    .collect::<Vec<_>>();
                let nonzero_source = packable(&nonzero, &[900], dtype);
                let mut nonzero_packed = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_packbits(nonzero_source, bit_order, &mut nonzero_packed),
                    0
                );
                assert_array(
                    nonzero_packed,
                    &[113],
                    &flat_pack_reference(&source_values, bit_order == 1),
                );
                super::super::native_support::delete_arrays(&[nonzero_source, nonzero_packed]);
            }
        }

        let bytes = (0..277).map(|value| value % 256).collect::<Vec<_>>();
        let byte_array = packable(&bytes, &[277], 7);
        let mut strided = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(byte_array, 0, 0, 277, 2, &mut strided),
            0
        );
        for source in [byte_array, strided] {
            let expected = super::super::native_support::ints(source).unwrap();
            let mut unpacked = std::ptr::null_mut();
            let mut repacked = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_unpackbits_all(source, 0, &mut unpacked),
                0
            );
            assert_eq!(terlan_ndarray_v1_packbits(unpacked, 0, &mut repacked), 0);
            assert_eq!(
                super::super::native_support::ints(repacked).unwrap(),
                expected
            );
            super::super::native_support::delete_arrays(&[unpacked, repacked]);
        }
        let tiled = bytes.repeat(3);
        let matrix = packable(&tiled, &[3, 277], 7);
        let mut unpacked_rows = std::ptr::null_mut();
        let mut repacked_rows = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_unpackbits_axis_all(matrix, 1, 0, &mut unpacked_rows),
            0
        );
        assert_eq!(
            terlan_ndarray_v1_packbits_axis(unpacked_rows, 1, 0, &mut repacked_rows),
            0
        );
        assert_eq!(
            super::super::native_support::ints(repacked_rows).unwrap(),
            tiled
        );
        let mut transposed = Vec::with_capacity(tiled.len());
        for column in 0..277 {
            for row in 0..3 {
                transposed.push(tiled[row * 277 + column]);
            }
        }
        let columns = packable(&transposed, &[277, 3], 7);
        let mut unpacked_columns = std::ptr::null_mut();
        let mut repacked_columns = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_unpackbits_axis_all(columns, 0, 0, &mut unpacked_columns),
            0
        );
        assert_eq!(
            terlan_ndarray_v1_packbits_axis(unpacked_columns, 0, 0, &mut repacked_columns),
            0
        );
        assert_eq!(
            super::super::native_support::ints(repacked_columns).unwrap(),
            transposed
        );
        super::super::native_support::delete_arrays(&[
            byte_array,
            strided,
            matrix,
            unpacked_rows,
            repacked_rows,
            columns,
            unpacked_columns,
            repacked_columns,
        ]);

        for width in 950..1050 {
            let ones = vec![1; 200 * width];
            for dtype in [1, 4, 7, 5, 8, 6, 9, 2, 10] {
                let source = packable(&ones, &[200, width as i64], dtype);
                let mut packed = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_packbits_axis(source, 1, 0, &mut packed),
                    0
                );
                assert_eq!(
                    super::super::native_support::shape(packed).unwrap(),
                    [200, ((width + 7) / 8) as i64]
                );
                super::super::native_support::delete_arrays(&[source, packed]);
            }
        }
    }
}

#[test]
fn unpackbit_counts_match_numpy_padding_trimming_axes_and_errors() {
    unsafe {
        let values = [
            1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1,
            0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
        ];
        let source = packable(&values, &[7, 7], 7);
        for bit_order in 0..=1 {
            let mut packed = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_packbits(source, bit_order, &mut packed),
                0
            );
            let mut padded = values.to_vec();
            padded.resize(57, 0);
            let mut all = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_unpackbits_all(packed, bit_order, &mut all),
                0
            );
            assert_array(all, &[56], &padded[..56]);
            super::super::native_support::delete_arrays(&[all]);
            for count in (0..58).chain((-56..=-1).rev()) {
                let mut unpacked = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_unpackbits(packed, count, bit_order, &mut unpacked),
                    0
                );
                let length = if count >= 0 {
                    count as usize
                } else {
                    (56 + count) as usize
                };
                assert_array(unpacked, &[length as i64], &padded[..length]);
                super::super::native_support::delete_arrays(&[unpacked]);
            }
            for axis in 0..=1 {
                let mut packed_axis = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_packbits_axis(source, axis, bit_order, &mut packed_axis),
                    0
                );
                for count in (0..8).chain((-8..=-1).rev()) {
                    let mut unpacked = std::ptr::null_mut();
                    assert_eq!(
                        terlan_ndarray_v1_unpackbits_axis(
                            packed_axis,
                            axis,
                            count,
                            bit_order,
                            &mut unpacked,
                        ),
                        0
                    );
                    let length = if count >= 0 { count } else { 8 + count };
                    let expected_shape = if axis == 0 {
                        vec![length, 7]
                    } else {
                        vec![7, length]
                    };
                    assert_eq!(
                        super::super::native_support::shape(unpacked).unwrap(),
                        expected_shape
                    );
                    assert_eq!(
                        super::super::native_support::ints(unpacked).unwrap(),
                        axis_prefix(&values, axis, length as usize)
                    );
                    super::super::native_support::delete_arrays(&[unpacked]);
                }
                let mut rejected = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_unpackbits_axis(
                        packed_axis,
                        axis,
                        -9,
                        bit_order,
                        &mut rejected
                    ),
                    INVALID_ARGUMENT
                );
                super::super::native_support::delete_arrays(&[packed_axis]);
            }
            super::super::native_support::delete_arrays(&[packed]);
        }
        for axis in 0..=1 {
            let mut packed_axis = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_packbits_axis(source, axis, 0, &mut packed_axis),
                0
            );
            for bit_order in 0..=1 {
                let mut unpacked = std::ptr::null_mut();
                assert_eq!(
                    terlan_ndarray_v1_unpackbits_axis_all(
                        packed_axis,
                        axis,
                        bit_order,
                        &mut unpacked,
                    ),
                    0
                );
                let expected_shape = if axis == 0 { [8, 7] } else { [7, 8] };
                assert_array(
                    unpacked,
                    &expected_shape,
                    &axis_all(&values, axis, bit_order == 1),
                );
                super::super::native_support::delete_arrays(&[unpacked]);
            }
            super::super::native_support::delete_arrays(&[packed_axis]);
        }
        let mut rejected = std::ptr::null_mut();
        let mut packed = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_packbits(source, 0, &mut packed), 0);
        assert_eq!(
            terlan_ndarray_v1_unpackbits(packed, -57, 0, &mut rejected),
            INVALID_ARGUMENT
        );
        assert!(rejected.is_null());
        super::super::native_support::delete_arrays(&[source, packed]);
    }
}

#[test]
fn data_completion_handles_all_nan_quantiles_and_rejects_bad_bit_requests() {
    unsafe {
        let nan = array(&[f64::NAN], &[1]);
        let mut index = 0;
        let mut output = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_nanargmax(nan, &mut index), DOMAIN_ERROR);
        assert_eq!(terlan_ndarray_v1_nanquantile(nan, 0.5, &mut output), 0);
        assert!(values(output)[0].is_nan());
        super::super::native_support::delete_arrays(&[output]);
        output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_packbits(nan, 7, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_unpackbits(nan, -2, 0, &mut output),
            DOMAIN_ERROR
        );
        assert!(output.is_null());
        super::super::native_support::delete_arrays(&[nan]);
    }
}
