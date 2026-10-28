//! Chebyshev polynomial C ABI operations.

use super::native_support::Array;
use super::polynomial_basis_support::{self as basis, Basis};

macro_rules! unary {
    ($rust:ident, $symbol:ident, $operation:ident) => {
        #[doc = concat!("Performs Chebyshev `", stringify!($operation), "`.")]
        #[no_mangle]
        pub unsafe extern "C" fn $symbol(value: Array, output: *mut Array) -> i32 {
            unsafe { basis::$rust(value, Basis::Chebyshev, output) }
        }
    };
}

macro_rules! binary {
    ($rust:ident, $symbol:ident, $operation:ident) => {
        #[doc = concat!("Performs Chebyshev `", stringify!($operation), "`.")]
        #[no_mangle]
        pub unsafe extern "C" fn $symbol(lhs: Array, rhs: Array, output: *mut Array) -> i32 {
            unsafe { basis::$rust(lhs, rhs, output) }
        }
    };
}

/// Converts Chebyshev coefficients to power-basis coefficients.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_cheb2poly(value: Array, output: *mut Array) -> i32 {
    unsafe { basis::convert(value, Basis::Chebyshev, true, output) }
}

/// Converts power-basis coefficients to Chebyshev coefficients.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_poly2cheb(value: Array, output: *mut Array) -> i32 {
    unsafe { basis::convert(value, Basis::Chebyshev, false, output) }
}

binary!(add, terlan_ndarray_v1_chebadd, addition);
binary!(subtract, terlan_ndarray_v1_chebsub, subtraction);

/// Multiplies two Chebyshev coefficient vectors.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebmul(
    lhs: Array,
    rhs: Array,
    output: *mut Array,
) -> i32 {
    unsafe { basis::multiply(lhs, rhs, Basis::Chebyshev, output) }
}

/// Divides two Chebyshev coefficient vectors into quotient and remainder.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebdiv(
    lhs: Array,
    rhs: Array,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    unsafe { basis::divide(lhs, rhs, Basis::Chebyshev, output, output_length) }
}

/// Multiplies Chebyshev coefficients by `x`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebmulx(value: Array, output: *mut Array) -> i32 {
    unsafe { basis::multiply_x(value, Basis::Chebyshev, output) }
}

/// Raises Chebyshev coefficients to an integer power.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebpow(
    value: Array,
    exponent: i64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::power(value, exponent, Basis::Chebyshev, output) }
}

/// Differentiates Chebyshev coefficients.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebder(
    value: Array,
    order: i64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::derivative(value, order, Basis::Chebyshev, output) }
}

/// Integrates Chebyshev coefficients.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebint(
    value: Array,
    order: i64,
    constant: f64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::integral(value, order, constant, Basis::Chebyshev, output) }
}

/// Fits Chebyshev coefficients to coordinate/value samples.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebfit(
    coordinates: Array,
    values: Array,
    degree: i64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::fit(coordinates, values, degree, Basis::Chebyshev, output) }
}

/// Builds Chebyshev coefficients from roots.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebfromroots(roots: Array, output: *mut Array) -> i32 {
    unsafe { basis::from_roots(roots, Basis::Chebyshev, output) }
}

unary!(companion, terlan_ndarray_v1_chebcompanion, companion);
unary!(roots, terlan_ndarray_v1_chebroots, roots);

/// Removes insignificant high-order Chebyshev coefficients.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebtrim(
    value: Array,
    tolerance: f64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::trim(value, tolerance, output) }
}

/// Returns Chebyshev coefficients for `offset + scale * x`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebline(
    offset: f64,
    scale: f64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::line(offset, scale, output) }
}

/// Evaluates a Chebyshev coefficient vector.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebval(
    coordinates: Array,
    coefficients: Array,
    output: *mut Array,
) -> i32 {
    unsafe {
        basis::evaluate(
            &[coordinates],
            coefficients,
            Basis::Chebyshev,
            false,
            output,
        )
    }
}

macro_rules! evaluate_nd {
    ($symbol:ident, $grid:expr, $($coordinate:ident),+) => {
        #[doc = "Evaluates a multidimensional Chebyshev coefficient tensor."]
        #[no_mangle]
        pub unsafe extern "C" fn $symbol(
            $($coordinate: Array,)+
            coefficients: Array,
            output: *mut Array,
        ) -> i32 {
            unsafe { basis::evaluate(&[$($coordinate),+], coefficients, Basis::Chebyshev, $grid, output) }
        }
    };
}

evaluate_nd!(terlan_ndarray_v1_chebval2d, false, x, y);
evaluate_nd!(terlan_ndarray_v1_chebval3d, false, x, y, z);
evaluate_nd!(terlan_ndarray_v1_chebgrid2d, true, x, y);
evaluate_nd!(terlan_ndarray_v1_chebgrid3d, true, x, y, z);

/// Builds a one-dimensional Chebyshev Vandermonde matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebvander(
    x: Array,
    degree: i64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::vander(&[x], &[degree], Basis::Chebyshev, output) }
}

/// Builds a two-dimensional Chebyshev Vandermonde matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebvander2d(
    x: Array,
    y: Array,
    x_degree: i64,
    y_degree: i64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::vander(&[x, y], &[x_degree, y_degree], Basis::Chebyshev, output) }
}

/// Builds a three-dimensional Chebyshev Vandermonde matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebvander3d(
    x: Array,
    y: Array,
    z: Array,
    x_degree: i64,
    y_degree: i64,
    z_degree: i64,
    output: *mut Array,
) -> i32 {
    unsafe {
        basis::vander(
            &[x, y, z],
            &[x_degree, y_degree, z_degree],
            Basis::Chebyshev,
            output,
        )
    }
}

/// Returns first-kind Chebyshev sample points.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebpts1(count: i64, output: *mut Array) -> i32 {
    unsafe { basis::chebyshev_points_first(count, output) }
}

/// Returns second-kind Chebyshev sample points.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebpts2(count: i64, output: *mut Array) -> i32 {
    unsafe { basis::chebyshev_points_second(count, output) }
}

/// Returns Chebyshev Gaussian quadrature nodes and weights.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebgauss(degree: i64, output: *mut Array) -> i32 {
    unsafe { basis::gauss(degree, Basis::Chebyshev, output) }
}

/// Evaluates the Chebyshev weight function.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebweight(
    coordinates: Array,
    output: *mut Array,
) -> i32 {
    unsafe { basis::weight(coordinates, Basis::Chebyshev, output) }
}

/// Interpolates values sampled at first-kind Chebyshev points.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_chebinterpolate(
    values: Array,
    output: *mut Array,
) -> i32 {
    unsafe { basis::chebyshev_interpolate(values, output) }
}

#[cfg(test)]
#[path = "ndarray_polynomial_chebyshev_tests.rs"]
mod tests;
