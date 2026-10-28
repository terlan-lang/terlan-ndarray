//! Extended NumPy orthogonal polynomial conformance tests.

use super::*;

#[test]
fn numpy_orthogonal_companion_and_degree_100_gauss_contracts_match() {
    unsafe {
        verify_companion_and_gauss(
            super::super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebcompanion,
            super::super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebgauss,
            super::super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebvander,
            std::f64::consts::PI,
        );
        verify_companion_and_gauss(
            super::super::super::polynomial_legendre_extension::terlan_ndarray_v1_legcompanion,
            super::super::super::polynomial_legendre_extension::terlan_ndarray_v1_leggauss,
            super::super::super::polynomial_legendre_extension::terlan_ndarray_v1_legvander,
            2.0,
        );
    }
}

#[test]
fn numpy_chebyshev_sampled_interpolation_reproduces_every_admitted_power() {
    unsafe {
        let coordinates = array(&[-1.0, -0.75, -0.5, 0.0, 0.25, 0.75, 1.0], &[7]);
        for degree in 0..10 {
            let count = degree + 1;
            let points = (0..count)
                .map(|index| {
                    let odd = -(count as i64) + 1 + 2 * index as i64;
                    (0.5 * std::f64::consts::PI / count as f64 * odd as f64).sin()
                })
                .collect::<Vec<_>>();
            for power in 0..=degree {
                let samples = points
                    .iter()
                    .map(|value| value.powi(power as i32))
                    .collect::<Vec<_>>();
                let sampled = array(&samples, &[count as i64]);
                let mut coefficients = std::ptr::null_mut();
                let mut evaluated = std::ptr::null_mut();
                assert_eq!(
                    super::super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebinterpolate(
                        sampled,
                        &mut coefficients,
                    ),
                    OK
                );
                assert_eq!(dimensions(coefficients), [count as i64]);
                assert_eq!(
                    super::super::super::polynomial_chebyshev_extension::terlan_ndarray_v1_chebval(
                        coordinates,
                        coefficients,
                        &mut evaluated,
                    ),
                    OK
                );
                close(
                    &values(evaluated),
                    &[-1.0_f64, -0.75, -0.5, 0.0, 0.25, 0.75, 1.0]
                        .iter()
                        .map(|value| value.powi(power as i32))
                        .collect::<Vec<_>>(),
                );
                dispose(&[evaluated, coefficients, sampled]);
            }
        }
        dispose(&[coordinates]);
    }
}
