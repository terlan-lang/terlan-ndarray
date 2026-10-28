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
fn legendre_converts_evaluates_and_finds_roots() {
    unsafe {
        let p2 = array(&[0.0, 0.0, 1.0], &[3]);
        let coordinates = array(&[-1.0, 0.0, 1.0], &[3]);
        let mut power = std::ptr::null_mut();
        let mut round_trip = std::ptr::null_mut();
        let mut evaluated = std::ptr::null_mut();
        let mut roots = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_leg2poly(p2, &mut power), 0);
        close(&values(power), &[-0.5, 0.0, 1.5]);
        assert_eq!(terlan_ndarray_v1_poly2leg(power, &mut round_trip), 0);
        close(&values(round_trip), &[0.0, 0.0, 1.0]);
        assert_eq!(terlan_ndarray_v1_legval(coordinates, p2, &mut evaluated), 0);
        close(&values(evaluated), &[1.0, -0.5, 1.0]);
        assert_eq!(terlan_ndarray_v1_legroots(p2, &mut roots), 0);
        let mut roots_values = real_parts(roots);
        roots_values.sort_by(f64::total_cmp);
        close(&roots_values, &[-1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt()]);
        super::super::native_support::delete_arrays(&[
            p2,
            coordinates,
            power,
            round_trip,
            evaluated,
            roots,
        ]);
    }
}

#[test]
fn legendre_gauss_integrates_quadratics_and_vander_uses_recurrence() {
    unsafe {
        let mut quadrature = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_leggauss(3, &mut quadrature), 0);
        let quadrature_values = values(quadrature);
        let (nodes, weights) = quadrature_values.split_at(3);
        let integral = nodes
            .iter()
            .zip(weights)
            .map(|(x, weight)| x * x * weight)
            .sum::<f64>();
        assert!((integral - 2.0 / 3.0).abs() < 1.0e-9);

        let coordinates = array(&[-1.0, 0.0, 1.0], &[3]);
        let mut vander = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_legvander(coordinates, 2, &mut vander), 0);
        close(
            &values(vander),
            &[1.0, -1.0, 1.0, 1.0, 0.0, -0.5, 1.0, 1.0, 1.0],
        );
        super::super::native_support::delete_arrays(&[quadrature, coordinates, vander]);
    }
}

#[test]
fn legendre_rejects_invalid_degrees_shapes_and_null_outputs() {
    unsafe {
        let vector = array(&[1.0, 2.0], &[2]);
        let mut output = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_legpow(vector, -1, &mut output), 100);
        assert_eq!(terlan_ndarray_v1_leggauss(0, std::ptr::null_mut()), 100);
        assert_eq!(
            terlan_ndarray_v1_legval2d(vector, vector, vector, &mut output),
            super::super::native_support::SHAPE_MISMATCH
        );
        assert!(output.is_null());
        super::super::native_support::delete_arrays(&[vector]);
    }
}
