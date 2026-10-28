//! NumPy 1.21.5 power-polynomial conformance tests.

use super::super::polynomial_power_extension as power;
use super::*;

const OK: i32 = 0;

/// Creates one owned Float64 fixture.
unsafe fn array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, dimensions, 3).unwrap() }
}

/// Reads one result as real values.
unsafe fn values(array: Array) -> Vec<f64> {
    unsafe {
        super::super::native_support::numeric(array, false)
            .unwrap()
            .values
            .into_iter()
            .map(|value| value.re)
            .collect()
    }
}

/// Returns one result shape.
unsafe fn dimensions(array: Array) -> Vec<i64> {
    unsafe { super::super::native_support::shape(array).unwrap() }
}

/// Deletes all supplied arrays.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { super::super::native_support::delete_arrays(arrays) }
}

/// Removes trailing exact zeros while preserving one coefficient.
fn trim(mut coefficients: Vec<f64>) -> Vec<f64> {
    while coefficients.len() > 1 && coefficients.last() == Some(&0.0) {
        coefficients.pop();
    }
    coefficients
}

/// Returns a low-to-high monomial coefficient vector.
fn monomial(degree: usize) -> Vec<f64> {
    let mut coefficients = vec![0.0; degree + 1];
    coefficients[degree] = 1.0;
    coefficients
}

/// Multiplies two low-to-high coefficient vectors in the test oracle.
fn multiply(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; lhs.len() + rhs.len() - 1];
    for (lhs_index, lhs_value) in lhs.iter().enumerate() {
        for (rhs_index, rhs_value) in rhs.iter().enumerate() {
            output[lhs_index + rhs_index] += lhs_value * rhs_value;
        }
    }
    trim(output)
}

/// Asserts approximate vector equality.
fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-7,
            "{actual} != {expected}"
        );
    }
}

/// Calls polynomial division and takes ownership of both returned values.
unsafe fn divide(lhs: &[f64], rhs: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32> {
    let lhs_array = unsafe { array(lhs, &[lhs.len() as i64]) };
    let rhs_array = unsafe { array(rhs, &[rhs.len() as i64]) };
    let mut outputs = std::ptr::null_mut();
    let mut output_length = 0usize;
    let status = unsafe {
        terlan_ndarray_v1_polydiv_all(lhs_array, rhs_array, &mut outputs, &mut output_length)
    };
    unsafe { dispose(&[lhs_array, rhs_array]) };
    if status != OK {
        return Err(status);
    }
    assert_eq!(output_length, 2);
    let handles = unsafe { std::slice::from_raw_parts(outputs, output_length) };
    let quotient = unsafe { values(handles[0] as usize as Array) };
    let remainder = unsafe { values(handles[1] as usize as Array) };
    for &handle in handles {
        unsafe { dispose(&[handle as usize as Array]) };
    }
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(outputs) };
    Ok((quotient, remainder))
}

#[test]
fn numpy_power_polynomial_arithmetic_matches_degree_first_loops() {
    unsafe {
        for i in 0..5 {
            for j in 0..5 {
                let lhs_values = monomial(i);
                let rhs_values = monomial(j);
                let lhs = array(&lhs_values, &[lhs_values.len() as i64]);
                let rhs = array(&rhs_values, &[rhs_values.len() as i64]);
                let mut added = std::ptr::null_mut();
                let mut subtracted = std::ptr::null_mut();
                let mut product = std::ptr::null_mut();
                assert_eq!(terlan_ndarray_v1_polyadd(lhs, rhs, &mut added), OK);
                assert_eq!(terlan_ndarray_v1_polysub(lhs, rhs, &mut subtracted), OK);
                assert_eq!(terlan_ndarray_v1_polymul(lhs, rhs, &mut product), OK);
                let mut sum = vec![0.0; i.max(j) + 1];
                sum[i] += 1.0;
                sum[j] += 1.0;
                let mut difference = vec![0.0; i.max(j) + 1];
                difference[i] += 1.0;
                difference[j] -= 1.0;
                close(&values(added), &trim(sum));
                close(&values(subtracted), &trim(difference));
                close(&values(product), &monomial(i + j));
                dispose(&[product, subtracted, added, rhs, lhs]);
            }
        }

        for degree in 0..5 {
            let coefficients = monomial(degree);
            let source = array(&coefficients, &[coefficients.len() as i64]);
            let mut output = std::ptr::null_mut();
            assert_eq!(power::terlan_ndarray_v1_polymulx(source, &mut output), OK);
            close(&values(output), &monomial(degree + 1));
            dispose(&[output, source]);
        }
        let zero = array(&[0.0], &[1]);
        let mut zero_x = std::ptr::null_mut();
        assert_eq!(power::terlan_ndarray_v1_polymulx(zero, &mut zero_x), OK);
        close(&values(zero_x), &[0.0]);
        dispose(&[zero_x, zero]);

        assert_eq!(divide(&[1.0], &[0.0]), Err(DOMAIN_ERROR));
        assert_eq!(divide(&[2.0], &[2.0]).unwrap(), (vec![1.0], vec![0.0]));
        assert_eq!(
            divide(&[2.0, 2.0], &[2.0]).unwrap(),
            (vec![1.0, 1.0], vec![0.0])
        );
        for i in 0..5 {
            for j in 0..5 {
                let mut ci = vec![0.0; i];
                ci.extend([1.0, 2.0]);
                let mut cj = vec![0.0; j];
                cj.extend([1.0, 2.0]);
                let mut target = vec![0.0; ci.len().max(cj.len())];
                for (index, value) in ci.iter().enumerate() {
                    target[index] += value;
                }
                for (index, value) in cj.iter().enumerate() {
                    target[index] += value;
                }
                let (quotient, remainder) = divide(&target, &ci).unwrap();
                let mut reconstructed = multiply(&quotient, &ci);
                reconstructed.resize(reconstructed.len().max(remainder.len()), 0.0);
                for (index, value) in remainder.iter().enumerate() {
                    reconstructed[index] += value;
                }
                close(&trim(reconstructed), &trim(target));
            }
        }

        for length in 1..=5 {
            let coefficients = (0..length).map(|value| value as f64).collect::<Vec<_>>();
            for exponent in 0..5 {
                let source = array(&coefficients, &[coefficients.len() as i64]);
                let mut output = std::ptr::null_mut();
                assert_eq!(
                    power::terlan_ndarray_v1_polypow(source, exponent, &mut output),
                    OK
                );
                let expected =
                    (0..exponent).fold(vec![1.0], |value, _| multiply(&value, &coefficients));
                close(&values(output), &expected);
                dispose(&[output, source]);
            }
        }
    }
}

#[test]
fn numpy_power_polynomial_evaluation_and_vandermonde_shapes_match() {
    unsafe {
        let coordinates_values = (0..50)
            .map(|index| -1.0 + 2.0 * index as f64 / 49.0)
            .collect::<Vec<_>>();
        let coordinates = array(&coordinates_values, &[50]);
        for degree in 0..5 {
            let coefficients_values = monomial(degree);
            let coefficients = array(&coefficients_values, &[coefficients_values.len() as i64]);
            let mut output = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_polyval(coordinates, coefficients, &mut output),
                OK
            );
            let expected = coordinates_values
                .iter()
                .map(|value| value.powi(degree as i32))
                .collect::<Vec<_>>();
            close(&values(output), &expected);
            dispose(&[output, coefficients]);
        }
        let cubic = array(&[0.0, -1.0, 0.0, 1.0], &[4]);
        let mut cubic_values = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyval(coordinates, cubic, &mut cubic_values),
            OK
        );
        close(
            &values(cubic_values),
            &coordinates_values
                .iter()
                .map(|value| value * (value * value - 1.0))
                .collect::<Vec<_>>(),
        );
        dispose(&[cubic_values, cubic, coordinates]);

        for shape in [vec![], vec![2], vec![2, 2]] {
            let count = shape.iter().product::<i64>().max(1) as usize;
            let x = array(&vec![0.0; count], &shape);
            for coefficients_values in [vec![1.0], vec![1.0, 0.0], vec![1.0, 0.0, 0.0]] {
                let coefficients = array(&coefficients_values, &[coefficients_values.len() as i64]);
                let mut output = std::ptr::null_mut();
                assert_eq!(terlan_ndarray_v1_polyval(x, coefficients, &mut output), OK);
                assert_eq!(dimensions(output), shape);
                dispose(&[output, coefficients]);
            }
            dispose(&[x]);
        }
        let empty = array(&[], &[0]);
        let one = array(&[1.0], &[1]);
        let mut empty_output = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_polyval(empty, one, &mut empty_output), OK);
        assert_eq!(dimensions(empty_output), [0]);
        dispose(&[empty_output, one, empty]);

        let x = array(&[0.0, 1.0, 2.0], &[3]);
        let mut vander = std::ptr::null_mut();
        assert_eq!(power::terlan_ndarray_v1_polyvander(x, 3, &mut vander), OK);
        assert_eq!(dimensions(vander), [3, 4]);
        close(
            &values(vander),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 8.0],
        );
        let matrix = array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let mut matrix_vander = std::ptr::null_mut();
        assert_eq!(
            power::terlan_ndarray_v1_polyvander(matrix, 3, &mut matrix_vander),
            OK
        );
        assert_eq!(dimensions(matrix_vander), [3, 2, 4]);
        close(
            &values(matrix_vander),
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 8.0, 1.0, 3.0, 9.0, 27.0, 1.0, 4.0, 16.0, 64.0,
                1.0, 5.0, 25.0, 125.0, 1.0, 6.0, 36.0, 216.0,
            ],
        );
        let mut invalid = std::ptr::null_mut();
        assert_eq!(
            power::terlan_ndarray_v1_polyvander(x, -1, &mut invalid),
            INVALID_ARGUMENT
        );
        assert!(invalid.is_null());
        dispose(&[matrix_vander, matrix, vander, x]);
    }
}

#[test]
fn numpy_power_polynomial_multivariate_evaluation_matches_tensor_products() {
    unsafe {
        let x_values = [-0.75, -0.25, 0.25, 0.75, 1.25];
        let y_values = [-0.5, 0.0, 0.5, 1.0, 1.5];
        let z_values = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let x = array(&x_values, &[5]);
        let y = array(&y_values, &[5]);
        let z = array(&z_values, &[5]);
        let c1 = [1.0, 2.0, 3.0];
        let c2 = c1
            .iter()
            .flat_map(|lhs| c1.iter().map(move |rhs| lhs * rhs))
            .collect::<Vec<_>>();
        let c3 = c2
            .iter()
            .flat_map(|lhs| c1.iter().map(move |rhs| lhs * rhs))
            .collect::<Vec<_>>();
        let coefficients_2d = array(&c2, &[3, 3]);
        let coefficients_3d = array(&c3, &[3, 3, 3]);
        let polynomial = |value: f64| 1.0 + 2.0 * value + 3.0 * value * value;

        let mut value_2d = std::ptr::null_mut();
        let mut value_3d = std::ptr::null_mut();
        assert_eq!(
            power::terlan_ndarray_v1_polyval2d(x, y, coefficients_2d, &mut value_2d),
            OK
        );
        assert_eq!(
            power::terlan_ndarray_v1_polyval3d(x, y, z, coefficients_3d, &mut value_3d),
            OK
        );
        close(
            &values(value_2d),
            &(0..5)
                .map(|index| polynomial(x_values[index]) * polynomial(y_values[index]))
                .collect::<Vec<_>>(),
        );
        close(
            &values(value_3d),
            &(0..5)
                .map(|index| {
                    polynomial(x_values[index])
                        * polynomial(y_values[index])
                        * polynomial(z_values[index])
                })
                .collect::<Vec<_>>(),
        );

        let mut grid_2d = std::ptr::null_mut();
        let mut grid_3d = std::ptr::null_mut();
        assert_eq!(
            power::terlan_ndarray_v1_polygrid2d(x, y, coefficients_2d, &mut grid_2d),
            OK
        );
        assert_eq!(
            power::terlan_ndarray_v1_polygrid3d(x, y, z, coefficients_3d, &mut grid_3d),
            OK
        );
        assert_eq!(dimensions(grid_2d), [5, 5]);
        assert_eq!(dimensions(grid_3d), [5, 5, 5]);
        close(
            &values(grid_2d),
            &x_values
                .iter()
                .flat_map(|x| {
                    y_values
                        .iter()
                        .map(move |y| polynomial(*x) * polynomial(*y))
                })
                .collect::<Vec<_>>(),
        );
        let mut expected_grid_3d = Vec::with_capacity(125);
        for x in x_values {
            for y in y_values {
                for z in z_values {
                    expected_grid_3d.push(polynomial(x) * polynomial(y) * polynomial(z));
                }
            }
        }
        close(&values(grid_3d), &expected_grid_3d);

        let shaped_coordinates = array(&[1.0; 6], &[2, 3]);
        let mut shaped_2d = std::ptr::null_mut();
        let mut shaped_3d = std::ptr::null_mut();
        let mut shaped_grid_2d = std::ptr::null_mut();
        let mut shaped_grid_3d = std::ptr::null_mut();
        assert_eq!(
            power::terlan_ndarray_v1_polyval2d(
                shaped_coordinates,
                shaped_coordinates,
                coefficients_2d,
                &mut shaped_2d,
            ),
            OK
        );
        assert_eq!(
            power::terlan_ndarray_v1_polyval3d(
                shaped_coordinates,
                shaped_coordinates,
                shaped_coordinates,
                coefficients_3d,
                &mut shaped_3d,
            ),
            OK
        );
        assert_eq!(
            power::terlan_ndarray_v1_polygrid2d(
                shaped_coordinates,
                shaped_coordinates,
                coefficients_2d,
                &mut shaped_grid_2d,
            ),
            OK
        );
        assert_eq!(
            power::terlan_ndarray_v1_polygrid3d(
                shaped_coordinates,
                shaped_coordinates,
                shaped_coordinates,
                coefficients_3d,
                &mut shaped_grid_3d,
            ),
            OK
        );
        assert_eq!(dimensions(shaped_2d), [2, 3]);
        assert_eq!(dimensions(shaped_3d), [2, 3]);
        assert_eq!(dimensions(shaped_grid_2d), [2, 3, 2, 3]);
        assert_eq!(dimensions(shaped_grid_3d), [2, 3, 2, 3, 2, 3]);

        let mut vander_2d = std::ptr::null_mut();
        let mut vander_3d = std::ptr::null_mut();
        assert_eq!(
            power::terlan_ndarray_v1_polyvander2d(x, y, 1, 2, &mut vander_2d),
            OK
        );
        assert_eq!(
            power::terlan_ndarray_v1_polyvander3d(x, y, z, 1, 2, 3, &mut vander_3d),
            OK
        );
        assert_eq!(dimensions(vander_2d), [5, 6]);
        assert_eq!(dimensions(vander_3d), [5, 24]);
        let wrapped_x = array(&x_values, &[1, 5]);
        let wrapped_y = array(&y_values, &[1, 5]);
        let wrapped_z = array(&z_values, &[1, 5]);
        let mut wrapped_vander_2d = std::ptr::null_mut();
        let mut wrapped_vander_3d = std::ptr::null_mut();
        assert_eq!(
            power::terlan_ndarray_v1_polyvander2d(
                wrapped_x,
                wrapped_y,
                1,
                2,
                &mut wrapped_vander_2d,
            ),
            OK
        );
        assert_eq!(
            power::terlan_ndarray_v1_polyvander3d(
                wrapped_x,
                wrapped_y,
                wrapped_z,
                1,
                2,
                3,
                &mut wrapped_vander_3d,
            ),
            OK
        );
        assert_eq!(dimensions(wrapped_vander_2d), [1, 5, 6]);
        assert_eq!(dimensions(wrapped_vander_3d), [1, 5, 24]);
        let vander_2d_values = values(vander_2d);
        let coefficients = [0.5, -1.0, 2.0, 3.0, -0.25, 0.75];
        let coefficients_array = array(&coefficients, &[2, 3]);
        let mut evaluated = std::ptr::null_mut();
        assert_eq!(
            power::terlan_ndarray_v1_polyval2d(x, y, coefficients_array, &mut evaluated),
            OK
        );
        let products = vander_2d_values
            .chunks_exact(6)
            .map(|row| {
                row.iter()
                    .zip(coefficients)
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum()
            })
            .collect::<Vec<f64>>();
        close(&values(evaluated), &products);

        let bad_y = array(&[1.0, 2.0], &[2]);
        let mut invalid = std::ptr::null_mut();
        assert_eq!(
            power::terlan_ndarray_v1_polyval2d(x, bad_y, coefficients_array, &mut invalid),
            SHAPE_MISMATCH
        );
        assert!(invalid.is_null());
        dispose(&[
            bad_y,
            wrapped_vander_3d,
            wrapped_vander_2d,
            wrapped_z,
            wrapped_y,
            wrapped_x,
            evaluated,
            coefficients_array,
            vander_3d,
            vander_2d,
            grid_3d,
            grid_2d,
            shaped_grid_3d,
            shaped_grid_2d,
            shaped_3d,
            shaped_2d,
            shaped_coordinates,
            value_3d,
            value_2d,
            coefficients_3d,
            coefficients_2d,
            z,
            y,
            x,
        ]);
    }
}

#[test]
fn numpy_power_polynomial_roots_companion_trim_and_line_match() {
    unsafe {
        let empty = array(&[], &[0]);
        let mut from_empty = std::ptr::null_mut();
        assert_eq!(
            power::terlan_ndarray_v1_polyfromroots(empty, &mut from_empty),
            OK
        );
        close(&values(from_empty), &[1.0]);

        let expected = [
            vec![1.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0, 2.0],
            vec![0.0, -3.0, 0.0, 4.0],
            vec![1.0, 0.0, -8.0, 0.0, 8.0],
        ];
        for (degree, expected) in expected.iter().enumerate().skip(1) {
            let roots = (0..degree)
                .map(|index| {
                    let sample = 2 * index + 1;
                    (-std::f64::consts::PI
                        + sample as f64 * std::f64::consts::PI / (2 * degree) as f64)
                        .cos()
                })
                .collect::<Vec<_>>();
            let roots_array = array(&roots, &[degree as i64]);
            let mut coefficients = std::ptr::null_mut();
            assert_eq!(
                power::terlan_ndarray_v1_polyfromroots(roots_array, &mut coefficients),
                OK
            );
            let scale = 2.0f64.powi(degree as i32 - 1);
            close(
                &values(coefficients)
                    .into_iter()
                    .map(|value| value * scale)
                    .collect::<Vec<_>>(),
                expected,
            );
            let mut roots_back = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_polyroots(coefficients, &mut roots_back),
                OK
            );
            let mut actual_roots = values(roots_back);
            actual_roots.sort_by(f64::total_cmp);
            let mut expected_roots = roots;
            expected_roots.sort_by(f64::total_cmp);
            close(&actual_roots, &expected_roots);
            dispose(&[roots_back, coefficients, roots_array]);
        }

        let linear = array(&[1.0, 2.0], &[2]);
        let mut roots = std::ptr::null_mut();
        let mut companion = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_polyroots(linear, &mut roots), OK);
        assert_eq!(terlan_ndarray_v1_polycompanion(linear, &mut companion), OK);
        close(&values(roots), &[-0.5]);
        close(&values(companion), &[-0.5]);
        for degree in 1..5 {
            let coefficients_values = monomial(degree);
            let coefficients = array(&coefficients_values, &[coefficients_values.len() as i64]);
            let mut matrix = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_polycompanion(coefficients, &mut matrix),
                OK
            );
            assert_eq!(dimensions(matrix), [degree as i64, degree as i64]);
            dispose(&[matrix, coefficients]);
        }

        let coefficients = array(&[2.0, -1.0, 1.0, 0.0], &[4]);
        for (tolerance, expected) in [
            (0.0, vec![2.0, -1.0, 1.0]),
            (1.0, vec![2.0]),
            (2.0, vec![0.0]),
        ] {
            let mut output = std::ptr::null_mut();
            assert_eq!(
                power::terlan_ndarray_v1_polytrim(coefficients, tolerance, &mut output),
                OK
            );
            close(&values(output), &expected);
            dispose(&[output]);
        }
        let mut invalid = std::ptr::null_mut();
        assert_eq!(
            power::terlan_ndarray_v1_polytrim(coefficients, -1.0, &mut invalid),
            INVALID_ARGUMENT
        );

        let mut line = std::ptr::null_mut();
        let mut constant = std::ptr::null_mut();
        assert_eq!(power::terlan_ndarray_v1_polyline(3.0, 4.0, &mut line), OK);
        assert_eq!(
            power::terlan_ndarray_v1_polyline(3.0, 0.0, &mut constant),
            OK
        );
        close(&values(line), &[3.0, 4.0]);
        close(&values(constant), &[3.0]);
        dispose(&[
            constant,
            line,
            coefficients,
            companion,
            roots,
            linear,
            from_empty,
            empty,
        ]);
    }
}
