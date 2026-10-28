use super::*;

/// Creates one owned Float64 fixture.
unsafe fn array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, dimensions, 3).unwrap() }
}

/// Reads one result as real values.
unsafe fn values(array: Array) -> Vec<f64> {
    unsafe { super::super::native_support::floats(array, false).unwrap() }
}

/// Asserts approximate vector equality.
fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-9,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn power_completion_builds_evaluates_and_transforms_coefficients() {
    unsafe {
        let roots = array(&[1.0, 2.0], &[2]);
        let mut from_roots = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_polyfromroots(roots, &mut from_roots), 0);
        close(&values(from_roots), &[2.0, -3.0, 1.0]);

        let mut line = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_polyline(2.0, 3.0, &mut line), 0);
        close(&values(line), &[2.0, 3.0]);

        let factor = array(&[-1.0, 1.0], &[2]);
        let mut multiplied = std::ptr::null_mut();
        let mut powered = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_polymulx(factor, &mut multiplied), 0);
        assert_eq!(terlan_ndarray_v1_polypow(factor, 2, &mut powered), 0);
        close(&values(multiplied), &[0.0, -1.0, 1.0]);
        close(&values(powered), &[1.0, -2.0, 1.0]);

        let padded = array(&[1.0, 2.0, 0.0, 0.0], &[4]);
        let mut trimmed = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_polytrim(padded, 0.0, &mut trimmed), 0);
        close(&values(trimmed), &[1.0, 2.0]);

        let coordinates = array(&[0.0, 1.0, 2.0], &[3]);
        let mut direct = std::ptr::null_mut();
        let mut vander = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyvalfromroots(coordinates, roots, &mut direct),
            0
        );
        assert_eq!(terlan_ndarray_v1_polyvander(coordinates, 2, &mut vander), 0);
        close(&values(direct), &[2.0, 0.0, 0.0]);
        close(
            &values(vander),
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0],
        );

        super::super::native_support::delete_arrays(&[
            roots,
            from_roots,
            line,
            factor,
            multiplied,
            powered,
            padded,
            trimmed,
            coordinates,
            direct,
            vander,
        ]);
    }
}

#[test]
fn multivariate_power_evaluation_broadcasts_grids_and_vandermonde_rows() {
    unsafe {
        let x = array(&[1.0, 2.0], &[2, 1]);
        let y = array(&[3.0, 4.0, 5.0], &[1, 3]);
        let coefficients_2d = array(&[0.0, 1.0, 1.0, 0.0], &[2, 2]);
        let mut pointwise = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyval2d(x, y, coefficients_2d, &mut pointwise),
            0
        );
        assert_eq!(shape(pointwise).unwrap(), [2, 3]);
        close(&values(pointwise), &[4.0, 5.0, 6.0, 5.0, 6.0, 7.0]);

        let grid_x = array(&[1.0, 2.0], &[2]);
        let grid_y = array(&[3.0, 4.0, 5.0], &[3]);
        let mut grid = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polygrid2d(grid_x, grid_y, coefficients_2d, &mut grid),
            0
        );
        assert_eq!(shape(grid).unwrap(), [2, 3]);
        close(&values(grid), &[4.0, 5.0, 6.0, 5.0, 6.0, 7.0]);

        let z = array(&[10.0, 20.0], &[2]);
        let xyz = array(&[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[2, 2, 2]);
        let mut grid_3d = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polygrid3d(grid_x, grid_y, z, xyz, &mut grid_3d),
            0
        );
        assert_eq!(shape(grid_3d).unwrap(), [2, 3, 2]);
        close(
            &values(grid_3d),
            &[
                14.0, 24.0, 15.0, 25.0, 16.0, 26.0, 15.0, 25.0, 16.0, 26.0, 17.0, 27.0,
            ],
        );

        let scalar_x = array(&[2.0], &[1]);
        let scalar_y = array(&[3.0], &[1]);
        let scalar_z = array(&[5.0], &[1]);
        let mut vander_2d = std::ptr::null_mut();
        let mut vander_3d = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polyvander2d(scalar_x, scalar_y, 1, 1, &mut vander_2d),
            0
        );
        assert_eq!(
            terlan_ndarray_v1_polyvander3d(scalar_x, scalar_y, scalar_z, 1, 1, 1, &mut vander_3d,),
            0
        );
        close(&values(vander_2d), &[1.0, 3.0, 2.0, 6.0]);
        close(
            &values(vander_3d),
            &[1.0, 5.0, 3.0, 15.0, 2.0, 10.0, 6.0, 30.0],
        );

        super::super::native_support::delete_arrays(&[
            x,
            y,
            coefficients_2d,
            pointwise,
            grid_x,
            grid_y,
            grid,
            z,
            xyz,
            grid_3d,
            scalar_x,
            scalar_y,
            scalar_z,
            vander_2d,
            vander_3d,
        ]);
    }
}

#[test]
fn power_completion_rejects_invalid_degrees_shapes_and_tolerances() {
    unsafe {
        let vector = array(&[1.0, 2.0], &[2]);
        let incompatible = array(&[1.0, 2.0, 3.0], &[3]);
        let bad_coefficients = array(&[1.0, 2.0], &[2]);
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_polypow(vector, -1, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_polytrim(vector, f64::NAN, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_polyval2d(vector, incompatible, bad_coefficients, &mut output),
            SHAPE_MISMATCH
        );
        assert_eq!(
            terlan_ndarray_v1_polyvander2d(vector, vector, -1, 2, &mut output),
            INVALID_ARGUMENT
        );
        assert!(output.is_null());
        super::super::native_support::delete_arrays(&[vector, incompatible, bad_coefficients]);
    }
}
