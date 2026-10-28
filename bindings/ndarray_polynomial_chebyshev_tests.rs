use super::*;

unsafe fn array(values: &[f64], dimensions: &[i64]) -> Array {
    unsafe { super::super::native_support::owned_real(values, dimensions, 3).unwrap() }
}

unsafe fn values(array: Array) -> Vec<f64> {
    unsafe { super::super::native_support::floats(array, false).unwrap() }
}

unsafe fn real_parts(array: Array) -> Vec<f64> {
    unsafe {
        super::super::native_support::numeric(array, true)
            .unwrap()
            .values
            .into_iter()
            .map(|value| value.re)
            .collect()
    }
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-8, "{actual} != {expected}");
    }
}

#[test]
fn chebyshev_converts_evaluates_and_builds_vandermonde_values() {
    unsafe {
        let t2 = array(&[0.0, 0.0, 1.0], &[3]);
        let coordinates = array(&[-1.0, 0.0, 0.5, 1.0], &[4]);
        let mut power = std::ptr::null_mut();
        let mut round_trip = std::ptr::null_mut();
        let mut evaluated = std::ptr::null_mut();
        let mut vander = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_cheb2poly(t2, &mut power), 0);
        close(&values(power), &[-1.0, 0.0, 2.0]);
        assert_eq!(terlan_ndarray_v1_poly2cheb(power, &mut round_trip), 0);
        close(&values(round_trip), &[0.0, 0.0, 1.0]);
        assert_eq!(
            terlan_ndarray_v1_chebval(coordinates, t2, &mut evaluated),
            0
        );
        close(&values(evaluated), &[1.0, -1.0, -0.5, 1.0]);
        assert_eq!(terlan_ndarray_v1_chebvander(coordinates, 2, &mut vander), 0);
        close(
            &values(vander),
            &[
                1.0, -1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.5, -0.5, 1.0, 1.0, 1.0,
            ],
        );
        super::super::native_support::delete_arrays(&[
            t2,
            coordinates,
            power,
            round_trip,
            evaluated,
            vander,
        ]);
    }
}

#[test]
fn chebyshev_roots_quadrature_points_and_multivariate_paths_work() {
    unsafe {
        let t2 = array(&[0.0, 0.0, 1.0], &[3]);
        let mut roots = std::ptr::null_mut();
        let mut points = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_chebroots(t2, &mut roots), 0);
        let mut roots_values = real_parts(roots);
        roots_values.sort_by(f64::total_cmp);
        close(&roots_values, &[-1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt()]);
        assert_eq!(terlan_ndarray_v1_chebpts1(4, &mut points), 0);
        assert_eq!(values(points).len(), 4);

        let mut quadrature = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_chebgauss(4, &mut quadrature), 0);
        assert_eq!(
            super::super::native_support::shape(quadrature).unwrap(),
            [2, 4]
        );
        close(&values(quadrature)[4..], &[std::f64::consts::PI / 4.0; 4]);

        let x = array(&[0.0, 0.5], &[2]);
        let y = array(&[-1.0, 1.0], &[2]);
        let coefficients = array(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let mut evaluated = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_chebval2d(x, y, coefficients, &mut evaluated),
            0
        );
        close(&values(evaluated), &[1.0, 1.5]);

        super::super::native_support::delete_arrays(&[
            t2,
            roots,
            points,
            quadrature,
            x,
            y,
            coefficients,
            evaluated,
        ]);
    }
}

#[test]
fn chebyshev_rejects_invalid_degrees_counts_and_shapes() {
    unsafe {
        let vector = array(&[1.0, 2.0], &[2]);
        let mut output = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_chebpow(vector, -1, &mut output), 100);
        assert_eq!(terlan_ndarray_v1_chebpts1(0, &mut output), 100);
        assert_eq!(terlan_ndarray_v1_chebpts2(1, &mut output), 100);
        assert_eq!(
            terlan_ndarray_v1_chebval2d(vector, vector, vector, &mut output),
            super::super::native_support::SHAPE_MISMATCH
        );
        assert!(output.is_null());
        super::super::native_support::delete_arrays(&[vector]);
    }
}
