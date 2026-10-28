//! NumPy 1.21.5 orthogonal polynomial tensor conformance tests.

use super::*;

/// Creates one owned Float64 fixture.
unsafe fn array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, dimensions, 3).unwrap() }
}

/// Creates one owned Int64 fixture.
unsafe fn int_array(values: &[i64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_int(values, dimensions).unwrap() }
}

/// Reads one result as real values.
unsafe fn values(array: Array) -> Vec<f64> {
    unsafe { super::super::native_support::floats(array, false).unwrap() }
}

/// Returns one result shape.
unsafe fn dimensions(array: Array) -> Vec<i64> {
    unsafe { shape(array).unwrap() }
}

/// Deletes all supplied arrays.
unsafe fn dispose(arrays: &[Array]) {
    unsafe { delete_arrays(arrays) }
}

/// Asserts approximate real equality.
fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-7,
            "{actual} != {expected}"
        );
    }
}

/// Binary coefficient operation used by the shared basis conformance harness.
type BinaryOperation = unsafe extern "C" fn(Array, Array, *mut Array) -> i32;

/// Coefficient division operation used by the shared basis conformance harness.
type DivideOperation = unsafe extern "C" fn(Array, Array, *mut *mut u64, *mut usize) -> i32;

/// Coefficient-by-x operation used by the shared basis conformance harness.
type UnaryOperation = unsafe extern "C" fn(Array, *mut Array) -> i32;

/// Integer coefficient-power operation used by the shared basis conformance harness.
type PowerOperation = unsafe extern "C" fn(Array, i64, *mut Array) -> i32;

/// One-dimensional basis evaluation operation.
type EvaluateOperation = unsafe extern "C" fn(Array, Array, *mut Array) -> i32;

/// Two-dimensional basis evaluation operation.
type Evaluate2Operation = unsafe extern "C" fn(Array, Array, Array, *mut Array) -> i32;

/// Three-dimensional basis evaluation operation.
type Evaluate3Operation = unsafe extern "C" fn(Array, Array, Array, Array, *mut Array) -> i32;

/// One-dimensional basis Vandermonde operation.
type VanderOperation = unsafe extern "C" fn(Array, i64, *mut Array) -> i32;

/// Two-dimensional basis Vandermonde operation.
type Vander2Operation = unsafe extern "C" fn(Array, Array, i64, i64, *mut Array) -> i32;

/// Three-dimensional basis Vandermonde operation.
type Vander3Operation = unsafe extern "C" fn(Array, Array, Array, i64, i64, i64, *mut Array) -> i32;

/// Coefficient conversion operation.
type ConvertOperation = unsafe extern "C" fn(Array, *mut Array) -> i32;

/// Root construction or extraction operation.
type RootOperation = unsafe extern "C" fn(Array, *mut Array) -> i32;

/// Companion-matrix operation.
type CompanionOperation = unsafe extern "C" fn(Array, *mut Array) -> i32;

/// Gaussian-quadrature operation.
type GaussOperation = unsafe extern "C" fn(i64, *mut Array) -> i32;

/// Creates a unit coefficient vector for one basis degree.
unsafe fn unit(degree: usize) -> Array {
    let mut coefficients = vec![0.0; degree + 1];
    coefficients[degree] = 1.0;
    unsafe { array(&coefficients, &[coefficients.len() as i64]) }
}

/// Trims insignificant trailing values with NumPy's minimum scalar length.
fn trimmed(mut values: Vec<f64>) -> Vec<f64> {
    while values.len() > 1 && values.last().is_some_and(|value| value.abs() <= 1.0e-12) {
        values.pop();
    }
    values
}

/// Takes ownership of arrays returned through one list-valued native operation.
unsafe fn returned_arrays(raw: *mut u64, length: usize) -> Vec<Array> {
    let arrays = unsafe { std::slice::from_raw_parts(raw, length) }
        .iter()
        .map(|identity| *identity as usize as Array)
        .collect::<Vec<_>>();
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(raw) };
    arrays
}

/// Reads one real or complex array through its real components.
unsafe fn real_values(array: Array) -> Vec<f64> {
    unsafe {
        super::super::native_support::numeric(array, true)
            .unwrap()
            .values
            .into_iter()
            .map(|value| value.re)
            .collect()
    }
}

/// Verifies NumPy's low-degree arithmetic identities for one orthogonal basis.
unsafe fn verify_arithmetic(
    add: BinaryOperation,
    subtract: BinaryOperation,
    multiply: BinaryOperation,
    divide: DivideOperation,
    multiply_x: UnaryOperation,
    power: PowerOperation,
    evaluate: EvaluateOperation,
) {
    let coordinates = unsafe { array(&[-1.0, -0.75, -0.25, 0.0, 0.5, 1.0], &[6]) };
    for left_degree in 0..5 {
        let left = unsafe { unit(left_degree) };
        let mut left_values = std::ptr::null_mut();
        assert_eq!(unsafe { evaluate(coordinates, left, &mut left_values) }, OK);
        let evaluated_left = unsafe { values(left_values) };

        let mut multiplied_x = std::ptr::null_mut();
        let mut evaluated_x = std::ptr::null_mut();
        assert_eq!(unsafe { multiply_x(left, &mut multiplied_x) }, OK);
        assert_eq!(
            unsafe { evaluate(coordinates, multiplied_x, &mut evaluated_x) },
            OK
        );
        close(
            &unsafe { values(evaluated_x) },
            &evaluated_left
                .iter()
                .zip([-1.0, -0.75, -0.25, 0.0, 0.5, 1.0])
                .map(|(value, coordinate)| value * coordinate)
                .collect::<Vec<_>>(),
        );

        for exponent in 0..5 {
            let mut powered = std::ptr::null_mut();
            let mut evaluated = std::ptr::null_mut();
            assert_eq!(unsafe { power(left, exponent, &mut powered) }, OK);
            assert_eq!(
                unsafe { evaluate(coordinates, powered, &mut evaluated) },
                OK
            );
            close(
                &unsafe { values(evaluated) },
                &evaluated_left
                    .iter()
                    .map(|value| value.powi(exponent as i32))
                    .collect::<Vec<_>>(),
            );
            unsafe { dispose(&[evaluated, powered]) };
        }

        for right_degree in 0..5 {
            let right = unsafe { unit(right_degree) };
            let mut added = std::ptr::null_mut();
            let mut subtracted = std::ptr::null_mut();
            let mut multiplied = std::ptr::null_mut();
            let mut right_values = std::ptr::null_mut();
            let mut product_values = std::ptr::null_mut();
            assert_eq!(unsafe { add(left, right, &mut added) }, OK);
            assert_eq!(unsafe { subtract(left, right, &mut subtracted) }, OK);
            assert_eq!(unsafe { multiply(left, right, &mut multiplied) }, OK);
            assert_eq!(
                unsafe { evaluate(coordinates, right, &mut right_values) },
                OK
            );
            assert_eq!(
                unsafe { evaluate(coordinates, multiplied, &mut product_values) },
                OK
            );

            let mut expected_add = vec![0.0; left_degree.max(right_degree) + 1];
            expected_add[left_degree] += 1.0;
            expected_add[right_degree] += 1.0;
            let mut expected_subtract = vec![0.0; left_degree.max(right_degree) + 1];
            expected_subtract[left_degree] += 1.0;
            expected_subtract[right_degree] -= 1.0;
            close(&unsafe { values(added) }, &trimmed(expected_add));
            close(&unsafe { values(subtracted) }, &trimmed(expected_subtract));
            close(
                &unsafe { values(product_values) },
                &evaluated_left
                    .iter()
                    .zip(unsafe { values(right_values) })
                    .map(|(lhs, rhs)| lhs * rhs)
                    .collect::<Vec<_>>(),
            );

            let mut outputs = std::ptr::null_mut();
            let mut output_length = 0usize;
            assert_eq!(
                unsafe { divide(added, left, &mut outputs, &mut output_length) },
                OK
            );
            assert_eq!(output_length, 2);
            let quotient_remainder = unsafe { returned_arrays(outputs, output_length) };
            let mut product = std::ptr::null_mut();
            let mut reconstructed = std::ptr::null_mut();
            let mut reconstructed_values = std::ptr::null_mut();
            assert_eq!(
                unsafe { multiply(quotient_remainder[0], left, &mut product) },
                OK
            );
            assert_eq!(
                unsafe { add(product, quotient_remainder[1], &mut reconstructed) },
                OK
            );
            assert_eq!(
                unsafe { evaluate(coordinates, reconstructed, &mut reconstructed_values) },
                OK
            );
            close(
                &unsafe { values(reconstructed_values) },
                &evaluated_left
                    .iter()
                    .zip(unsafe { values(right_values) })
                    .map(|(lhs, rhs)| lhs + rhs)
                    .collect::<Vec<_>>(),
            );
            unsafe {
                dispose(&[
                    reconstructed_values,
                    reconstructed,
                    product,
                    quotient_remainder[1],
                    quotient_remainder[0],
                    product_values,
                    right_values,
                    multiplied,
                    subtracted,
                    added,
                    right,
                ])
            };
        }
        unsafe { dispose(&[evaluated_x, multiplied_x, left_values, left]) };
    }
    unsafe { dispose(&[coordinates]) };
}

/// Verifies pointwise, Cartesian, and Vandermonde contracts for one basis.
unsafe fn verify_multivariate(
    coefficients_1d: &[f64],
    evaluate2d: Evaluate2Operation,
    evaluate3d: Evaluate3Operation,
    grid2d: Evaluate2Operation,
    grid3d: Evaluate3Operation,
    vander: VanderOperation,
    vander2d: Vander2Operation,
    vander3d: Vander3Operation,
) {
    let polynomial = |coordinate: f64| 1.0 + 2.0 * coordinate + 3.0 * coordinate * coordinate;
    let coefficients_2d = coefficients_1d
        .iter()
        .flat_map(|lhs| coefficients_1d.iter().map(move |rhs| lhs * rhs))
        .collect::<Vec<_>>();
    let coefficients_3d = coefficients_2d
        .iter()
        .flat_map(|lhs| coefficients_1d.iter().map(move |rhs| lhs * rhs))
        .collect::<Vec<_>>();
    let c2 = unsafe { array(&coefficients_2d, &[3, 3]) };
    let c3 = unsafe { array(&coefficients_3d, &[3, 3, 3]) };
    let x = unsafe { array(&[-0.8, 0.1, 0.7], &[3]) };
    let y = unsafe { array(&[-0.5, 0.2, 0.9], &[3]) };
    let z = unsafe { array(&[-0.3, 0.4, 0.8], &[3]) };
    let mut value2d = std::ptr::null_mut();
    let mut value3d = std::ptr::null_mut();
    assert_eq!(unsafe { evaluate2d(x, y, c2, &mut value2d) }, OK);
    assert_eq!(unsafe { evaluate3d(x, y, z, c3, &mut value3d) }, OK);
    close(
        &unsafe { values(value2d) },
        &[-0.8, 0.1, 0.7]
            .iter()
            .zip([-0.5, 0.2, 0.9])
            .map(|(x, y)| polynomial(*x) * polynomial(y))
            .collect::<Vec<_>>(),
    );
    close(
        &unsafe { values(value3d) },
        &[-0.8, 0.1, 0.7]
            .iter()
            .zip([-0.5, 0.2, 0.9])
            .zip([-0.3, 0.4, 0.8])
            .map(|((x, y), z)| polynomial(*x) * polynomial(y) * polynomial(z))
            .collect::<Vec<_>>(),
    );

    let grid_x = unsafe { array(&[-0.75, 0.5], &[2]) };
    let grid_y = unsafe { array(&[-0.5, 0.0, 0.75], &[3]) };
    let grid_z = unsafe { array(&[-0.25, 1.0], &[2]) };
    let mut cartesian2d = std::ptr::null_mut();
    let mut cartesian3d = std::ptr::null_mut();
    assert_eq!(unsafe { grid2d(grid_x, grid_y, c2, &mut cartesian2d) }, OK);
    assert_eq!(
        unsafe { grid3d(grid_x, grid_y, grid_z, c3, &mut cartesian3d) },
        OK
    );
    assert_eq!(unsafe { dimensions(cartesian2d) }, [2, 3]);
    assert_eq!(unsafe { dimensions(cartesian3d) }, [2, 3, 2]);
    let expected2d = [-0.75, 0.5]
        .iter()
        .flat_map(|x| {
            [-0.5, 0.0, 0.75]
                .iter()
                .map(move |y| polynomial(*x) * polynomial(*y))
        })
        .collect::<Vec<_>>();
    let expected3d = [-0.75, 0.5]
        .iter()
        .flat_map(|x| {
            [-0.5, 0.0, 0.75].iter().flat_map(move |y| {
                [-0.25, 1.0]
                    .iter()
                    .map(move |z| polynomial(*x) * polynomial(*y) * polynomial(*z))
            })
        })
        .collect::<Vec<_>>();
    close(&unsafe { values(cartesian2d) }, &expected2d);
    close(&unsafe { values(cartesian3d) }, &expected3d);

    let matrix_coordinates = unsafe { array(&[1.0, 2.0, 3.0, 4.0], &[2, 2]) };
    let mut matrix_vander = std::ptr::null_mut();
    assert_eq!(
        unsafe { vander(matrix_coordinates, 3, &mut matrix_vander) },
        OK
    );
    assert_eq!(unsafe { dimensions(matrix_vander) }, [2, 2, 4]);

    let mut design2d = std::ptr::null_mut();
    let mut design3d = std::ptr::null_mut();
    assert_eq!(unsafe { vander2d(x, y, 2, 2, &mut design2d) }, OK);
    assert_eq!(unsafe { vander3d(x, y, z, 2, 2, 2, &mut design3d) }, OK);
    assert_eq!(unsafe { dimensions(design2d) }, [3, 9]);
    assert_eq!(unsafe { dimensions(design3d) }, [3, 27]);
    let design2d_values = unsafe { values(design2d) };
    let design3d_values = unsafe { values(design3d) };
    for row in 0..3 {
        let projected2d = design2d_values[row * 9..(row + 1) * 9]
            .iter()
            .zip(&coefficients_2d)
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f64>();
        let projected3d = design3d_values[row * 27..(row + 1) * 27]
            .iter()
            .zip(&coefficients_3d)
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f64>();
        assert!((projected2d - unsafe { values(value2d) }[row]).abs() <= 1.0e-8);
        assert!((projected3d - unsafe { values(value3d) }[row]).abs() <= 1.0e-8);
    }
    unsafe {
        dispose(&[
            design3d,
            design2d,
            matrix_vander,
            matrix_coordinates,
            cartesian3d,
            cartesian2d,
            grid_z,
            grid_y,
            grid_x,
            value3d,
            value2d,
            z,
            y,
            x,
            c3,
            c2,
        ])
    };
}

/// Verifies companion dimensions and degree-100 quadrature orthogonality.
unsafe fn verify_companion_and_gauss(
    companion: CompanionOperation,
    gauss: GaussOperation,
    vander: VanderOperation,
    expected_weight_sum: f64,
) {
    for degree in 1..5 {
        let coefficients = unsafe { unit(degree) };
        let mut matrix = std::ptr::null_mut();
        assert_eq!(unsafe { companion(coefficients, &mut matrix) }, OK);
        assert_eq!(
            unsafe { dimensions(matrix) },
            [degree as i64, degree as i64]
        );
        unsafe { dispose(&[matrix, coefficients]) };
    }
    let linear = unsafe { array(&[1.0, 2.0], &[2]) };
    let mut linear_matrix = std::ptr::null_mut();
    assert_eq!(unsafe { companion(linear, &mut linear_matrix) }, OK);
    close(&unsafe { values(linear_matrix) }, &[-0.5]);

    let mut quadrature = std::ptr::null_mut();
    assert_eq!(unsafe { gauss(100, &mut quadrature) }, OK);
    let quadrature_values = unsafe { values(quadrature) };
    let (nodes, weights) = quadrature_values.split_at(100);
    assert!((weights.iter().sum::<f64>() - expected_weight_sum).abs() <= 1.0e-10);
    let nodes_array = unsafe { array(nodes, &[100]) };
    let mut design = std::ptr::null_mut();
    assert_eq!(unsafe { vander(nodes_array, 99, &mut design) }, OK);
    let design_values = unsafe { values(design) };
    let mut diagonal = vec![0.0; 100];
    for column in 0..100 {
        diagonal[column] = (0..100)
            .map(|row| {
                let value = design_values[row * 100 + column];
                weights[row] * value * value
            })
            .sum::<f64>();
    }
    for left in 0..100 {
        for right in 0..100 {
            let gram = (0..100)
                .map(|row| {
                    weights[row]
                        * design_values[row * 100 + left]
                        * design_values[row * 100 + right]
                })
                .sum::<f64>()
                / (diagonal[left] * diagonal[right]).sqrt();
            let expected = if left == right { 1.0 } else { 0.0 };
            assert!(
                (gram - expected).abs() <= 1.0e-7,
                "gram[{left},{right}]={gram}"
            );
        }
    }
    unsafe { dispose(&[design, nodes_array, quadrature, linear_matrix, linear]) };
}

#[test]
fn numpy_chebyshev_calculus_supports_axes_scaling_constants_and_bounds() {
    unsafe {
        let coefficients = array(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        );
        let constants = array(&[3.0], &[1]);
        let mut derivative = std::ptr::null_mut();
        let mut integral = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_chebder_axis(coefficients, 1, 0.5, 1, &mut derivative),
            OK
        );
        assert_eq!(dimensions(derivative), [3, 3]);
        close(
            &values(derivative),
            &[7.0, 6.0, 12.0, 15.0, 14.0, 24.0, 23.0, 22.0, 36.0],
        );
        assert_eq!(
            terlan_ndarray_v1_chebint_axis(
                coefficients,
                1,
                constants,
                -1.0,
                2.0,
                -1,
                &mut integral,
            ),
            OK
        );
        assert_eq!(dimensions(integral), [3, 5]);
        close(
            &values(integral),
            &[
                3.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                22.0 / 3.0,
                3.0,
                -1.0,
                7.0 / 3.0,
                2.0,
                35.0 / 3.0,
                7.0,
                -1.0,
                11.0 / 3.0,
                3.0,
            ],
        );
        dispose(&[integral, derivative, constants, coefficients]);
    }
}

#[test]
fn numpy_legendre_calculus_supports_axes_scaling_constants_and_bounds() {
    unsafe {
        let coefficients = array(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        );
        let constants = array(&[3.0], &[1]);
        let mut derivative = std::ptr::null_mut();
        let mut integral = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_legder_axis(coefficients, 1, 0.5, 1, &mut derivative),
            OK
        );
        assert_eq!(dimensions(derivative), [3, 3]);
        close(
            &values(derivative),
            &[3.0, 4.5, 10.0, 7.0, 10.5, 20.0, 11.0, 16.5, 30.0],
        );
        assert_eq!(
            terlan_ndarray_v1_legint_axis(coefficients, 1, constants, -1.0, 2.0, -1, &mut integral,),
            OK
        );
        assert_eq!(dimensions(integral), [3, 5]);
        close(
            &values(integral),
            &[
                11.0 / 3.0,
                0.8,
                4.0 / 21.0,
                1.2,
                8.0 / 7.0,
                9.0,
                7.2,
                12.0 / 7.0,
                2.8,
                16.0 / 7.0,
                43.0 / 3.0,
                13.6,
                68.0 / 21.0,
                4.4,
                24.0 / 7.0,
            ],
        );
        dispose(&[integral, derivative, constants, coefficients]);
    }
}

#[test]
fn numpy_orthogonal_evaluation_supports_tensor_and_broadcast_modes() {
    unsafe {
        let coordinates = array(&[0.0, 0.5], &[2]);
        let coefficients = array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let mut chebyshev_tensor = std::ptr::null_mut();
        let mut chebyshev_pointwise = std::ptr::null_mut();
        let mut legendre_tensor = std::ptr::null_mut();
        let mut legendre_pointwise = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_chebval_tensor(coordinates, coefficients, 1, &mut chebyshev_tensor,),
            OK
        );
        assert_eq!(
            terlan_ndarray_v1_chebval_tensor(
                coordinates,
                coefficients,
                0,
                &mut chebyshev_pointwise,
            ),
            OK
        );
        assert_eq!(
            terlan_ndarray_v1_legval_tensor(coordinates, coefficients, 1, &mut legendre_tensor,),
            OK
        );
        assert_eq!(
            terlan_ndarray_v1_legval_tensor(coordinates, coefficients, 0, &mut legendre_pointwise,),
            OK
        );
        assert_eq!(dimensions(chebyshev_tensor), [2, 2]);
        assert_eq!(dimensions(legendre_tensor), [2, 2]);
        close(&values(chebyshev_tensor), &[-4.0, 0.0, -4.0, 1.0]);
        close(&values(chebyshev_pointwise), &[-4.0, 1.0]);
        close(&values(legendre_tensor), &[-1.5, 1.875, -1.0, 3.25]);
        close(&values(legendre_pointwise), &[-1.5, 3.25]);
        dispose(&[
            legendre_pointwise,
            legendre_tensor,
            chebyshev_pointwise,
            chebyshev_tensor,
            coefficients,
            coordinates,
        ]);
    }
}

#[test]
fn numpy_orthogonal_fits_support_selected_degrees_weights_and_matrix_values() {
    unsafe {
        let x_values = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
        let even_values = x_values
            .iter()
            .map(|x| 1.0 + x * x + x * x * x * x)
            .collect::<Vec<_>>();
        let matrix_values = even_values
            .iter()
            .flat_map(|value| [*value, *value])
            .collect::<Vec<_>>();
        let coordinates = array(&x_values, &[9]);
        let observations = array(&matrix_values, &[9, 2]);
        let degrees = int_array(&[4, 2, 0], &[3]);
        let mut chebyshev = std::ptr::null_mut();
        let mut legendre = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_chebfit_degrees(coordinates, observations, degrees, &mut chebyshev,),
            OK
        );
        assert_eq!(
            terlan_ndarray_v1_legfit_degrees(coordinates, observations, degrees, &mut legendre),
            OK
        );
        close(
            &values(chebyshev),
            &[1.875, 1.875, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.125, 0.125],
        );
        close(
            &values(legendre),
            &[
                23.0 / 15.0,
                23.0 / 15.0,
                0.0,
                0.0,
                26.0 / 21.0,
                26.0 / 21.0,
                0.0,
                0.0,
                8.0 / 35.0,
                8.0 / 35.0,
            ],
        );

        let cubic = x_values
            .iter()
            .map(|x| x * (x - 1.0) * (x - 2.0))
            .collect::<Vec<_>>();
        let cubic_values = array(&cubic, &[9]);
        let all_degrees = int_array(&[0, 1, 2, 3], &[4]);
        let weights = array(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[9]);
        let mut weighted_chebyshev = std::ptr::null_mut();
        let mut weighted_legendre = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_chebfit_weighted_degrees(
                coordinates,
                cubic_values,
                all_degrees,
                weights,
                &mut weighted_chebyshev,
            ),
            OK
        );
        assert_eq!(
            terlan_ndarray_v1_legfit_weighted_degrees(
                coordinates,
                cubic_values,
                all_degrees,
                weights,
                &mut weighted_legendre,
            ),
            OK
        );
        close(&values(weighted_chebyshev), &[-1.5, 2.75, -1.5, 0.25]);
        close(&values(weighted_legendre), &[-1.0, 2.6, -2.0, 0.4]);
        dispose(&[
            weighted_legendre,
            weighted_chebyshev,
            weights,
            all_degrees,
            cubic_values,
            legendre,
            chebyshev,
            degrees,
            observations,
            coordinates,
        ]);
    }
}

#[test]
fn numpy_orthogonal_completion_rejects_invalid_axes_modes_and_degrees() {
    unsafe {
        let vector = array(&[1.0, 2.0], &[2]);
        let constants = array(&[], &[0]);
        let float_degrees = array(&[0.0, 1.0], &[2]);
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_chebder_axis(vector, 1, 1.0, 2, &mut output),
            super::super::native_support::AXIS_OUT_OF_RANGE
        );
        assert_eq!(
            terlan_ndarray_v1_legint_axis(vector, -1, constants, 0.0, 1.0, 0, &mut output,),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_chebval_tensor(vector, vector, 2, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_legfit_degrees(vector, vector, float_degrees, &mut output),
            super::super::native_support::UNSUPPORTED_DTYPE
        );
        assert!(output.is_null());
        dispose(&[float_degrees, constants, vector]);
    }
}

/// Verifies exact basis conversion for every canonical degree through nine.
unsafe fn verify_conversion(
    to_power: ConvertOperation,
    from_power: ConvertOperation,
    expected_power: &[&[f64]],
) {
    for (degree, expected) in expected_power.iter().enumerate() {
        let coefficients = unsafe { unit(degree) };
        let mut power = std::ptr::null_mut();
        let mut round_trip = std::ptr::null_mut();
        assert_eq!(unsafe { to_power(coefficients, &mut power) }, OK);
        close(&unsafe { values(power) }, expected);
        assert_eq!(unsafe { from_power(power, &mut round_trip) }, OK);
        close(
            &unsafe { values(round_trip) },
            &(0..=degree)
                .map(|index| if index == degree { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
        );
        unsafe { dispose(&[round_trip, power, coefficients]) };
    }
}

#[test]
fn numpy_orthogonal_arithmetic_matches_low_degree_basis_identities() {
    unsafe {
        verify_arithmetic(
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebadd,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebsub,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebmul,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebdiv,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebmulx,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebpow,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebval,
        );
        verify_arithmetic(
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legadd,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legsub,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legmul,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legdiv,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legmulx,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legpow,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legval,
        );
    }
}

#[test]
fn numpy_orthogonal_multivariate_evaluation_and_vandermonde_contracts_match() {
    unsafe {
        verify_multivariate(
            &[2.5, 2.0, 1.5],
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebval2d,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebval3d,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebgrid2d,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebgrid3d,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebvander,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebvander2d,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebvander3d,
        );
        verify_multivariate(
            &[2.0, 2.0, 2.0],
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legval2d,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legval3d,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_leggrid2d,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_leggrid3d,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legvander,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legvander2d,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legvander3d,
        );
    }
}

#[test]
fn numpy_orthogonal_conversion_matches_canonical_degrees_through_nine() {
    let chebyshev: &[&[f64]] = &[
        &[1.0],
        &[0.0, 1.0],
        &[-1.0, 0.0, 2.0],
        &[0.0, -3.0, 0.0, 4.0],
        &[1.0, 0.0, -8.0, 0.0, 8.0],
        &[0.0, 5.0, 0.0, -20.0, 0.0, 16.0],
        &[-1.0, 0.0, 18.0, 0.0, -48.0, 0.0, 32.0],
        &[0.0, -7.0, 0.0, 56.0, 0.0, -112.0, 0.0, 64.0],
        &[1.0, 0.0, -32.0, 0.0, 160.0, 0.0, -256.0, 0.0, 128.0],
        &[0.0, 9.0, 0.0, -120.0, 0.0, 432.0, 0.0, -576.0, 0.0, 256.0],
    ];
    let legendre: &[&[f64]] = &[
        &[1.0],
        &[0.0, 1.0],
        &[-0.5, 0.0, 1.5],
        &[0.0, -1.5, 0.0, 2.5],
        &[0.375, 0.0, -3.75, 0.0, 4.375],
        &[0.0, 1.875, 0.0, -8.75, 0.0, 7.875],
        &[-0.3125, 0.0, 6.5625, 0.0, -19.6875, 0.0, 14.4375],
        &[0.0, -2.1875, 0.0, 19.6875, 0.0, -43.3125, 0.0, 26.8125],
        &[
            0.2734375, 0.0, -9.84375, 0.0, 54.140625, 0.0, -93.84375, 0.0, 50.2734375,
        ],
        &[
            0.0, 2.4609375, 0.0, -36.09375, 0.0, 140.765625, 0.0, -201.09375, 0.0, 94.9609375,
        ],
    ];
    unsafe {
        verify_conversion(
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_cheb2poly,
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_poly2cheb,
            chebyshev,
        );
        verify_conversion(
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_leg2poly,
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_poly2leg,
            legendre,
        );
    }
}

#[test]
fn numpy_orthogonal_roots_trim_line_weight_and_points_match() {
    unsafe {
        let roots = array(&[-1.0, -0.3, 0.2, 1.0], &[4]);
        let coefficients = array(&[2.0, -1.0, 1.0, 0.0], &[4]);
        for (from_roots, extract_roots, trim, weight, legendre) in [
            (
                super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebfromroots
                    as RootOperation,
                super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebroots
                    as RootOperation,
                super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebtrim
                    as unsafe extern "C" fn(Array, f64, *mut Array) -> i32,
                super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebweight
                    as UnaryOperation,
                false,
            ),
            (
                super::super::polynomial_legendre_extension::terlan_ndarray_v1_legfromroots
                    as RootOperation,
                super::super::polynomial_legendre_extension::terlan_ndarray_v1_legroots
                    as RootOperation,
                super::super::polynomial_legendre_extension::terlan_ndarray_v1_legtrim
                    as unsafe extern "C" fn(Array, f64, *mut Array) -> i32,
                super::super::polynomial_legendre_extension::terlan_ndarray_v1_legweight
                    as UnaryOperation,
                true,
            ),
        ] {
            let mut polynomial = std::ptr::null_mut();
            let mut recovered = std::ptr::null_mut();
            let mut trimmed_zero = std::ptr::null_mut();
            let mut trimmed_one = std::ptr::null_mut();
            let weight_coordinates = array(&[-0.8, -0.4, 0.0, 0.4, 0.8], &[5]);
            let mut weights = std::ptr::null_mut();
            assert_eq!(from_roots(roots, &mut polynomial), OK);
            assert_eq!(extract_roots(polynomial, &mut recovered), OK);
            let mut recovered_values = real_values(recovered);
            recovered_values.sort_by(f64::total_cmp);
            close(&recovered_values, &[-1.0, -0.3, 0.2, 1.0]);
            assert_eq!(trim(coefficients, 0.0, &mut trimmed_zero), OK);
            assert_eq!(trim(coefficients, 1.0, &mut trimmed_one), OK);
            close(&values(trimmed_zero), &[2.0, -1.0, 1.0]);
            close(&values(trimmed_one), &[2.0]);
            assert_eq!(weight(weight_coordinates, &mut weights), OK);
            let expected_weights = [-0.8_f64, -0.4, 0.0, 0.4, 0.8]
                .iter()
                .map(|coordinate| {
                    if legendre {
                        1.0
                    } else {
                        1.0 / (1.0 - coordinate * coordinate).sqrt()
                    }
                })
                .collect::<Vec<_>>();
            close(&values(weights), &expected_weights);
            dispose(&[
                weights,
                weight_coordinates,
                trimmed_one,
                trimmed_zero,
                recovered,
                polynomial,
            ]);
        }
        let mut cheb_line = std::ptr::null_mut();
        let mut leg_line = std::ptr::null_mut();
        let mut leg_constant = std::ptr::null_mut();
        assert_eq!(
            super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebline(
                3.0,
                4.0,
                &mut cheb_line
            ),
            OK
        );
        assert_eq!(
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legline(
                3.0,
                4.0,
                &mut leg_line
            ),
            OK
        );
        assert_eq!(
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legline(
                3.0,
                0.0,
                &mut leg_constant
            ),
            OK
        );
        close(&values(cheb_line), &[3.0, 4.0]);
        close(&values(leg_line), &[3.0, 4.0]);
        close(&values(leg_constant), &[3.0]);
        for count in 1..=4 {
            let mut points = std::ptr::null_mut();
            assert_eq!(
                super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebpts1(
                    count,
                    &mut points
                ),
                OK
            );
            assert_eq!(values(points).len(), count as usize);
            dispose(&[points]);
        }
        for count in 2..=5 {
            let mut points = std::ptr::null_mut();
            assert_eq!(
                super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebpts2(
                    count,
                    &mut points
                ),
                OK
            );
            assert_eq!(values(points).len(), count as usize);
            dispose(&[points]);
        }
        let leg_coefficients = array(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let mut high_derivative = std::ptr::null_mut();
        let mut zero_integral = std::ptr::null_mut();
        assert_eq!(
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legder(
                leg_coefficients,
                4,
                &mut high_derivative
            ),
            OK
        );
        assert_eq!(
            super::super::polynomial_legendre_extension::terlan_ndarray_v1_legint(
                leg_coefficients,
                0,
                99.0,
                &mut zero_integral
            ),
            OK
        );
        close(&values(high_derivative), &[0.0]);
        close(&values(zero_integral), &[1.0, 2.0, 3.0, 4.0]);
        dispose(&[
            zero_integral,
            high_derivative,
            leg_coefficients,
            leg_constant,
            leg_line,
            cheb_line,
            coefficients,
            roots,
        ]);
    }
}

#[path = "ndarray_polynomial_basis_numpy_extended_tests.rs"]
mod extended;
