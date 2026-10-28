//! Legendre polynomial C ABI operations.

use super::native_support::Array;
use super::polynomial_basis_support::{self as basis, Basis};

macro_rules! unary {
    ($rust:ident, $symbol:ident, $operation:ident) => {
        #[doc = concat!("Performs Legendre `", stringify!($operation), "`.")]
        #[no_mangle]
        pub unsafe extern "C" fn $symbol(value: Array, output: *mut Array) -> i32 {
            unsafe { basis::$rust(value, Basis::Legendre, output) }
        }
    };
}

macro_rules! binary {
    ($rust:ident, $symbol:ident, $operation:ident) => {
        #[doc = concat!("Performs Legendre `", stringify!($operation), "`.")]
        #[no_mangle]
        pub unsafe extern "C" fn $symbol(lhs: Array, rhs: Array, output: *mut Array) -> i32 {
            unsafe { basis::$rust(lhs, rhs, output) }
        }
    };
}

/// Converts Legendre coefficients to power-basis coefficients.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_leg2poly(value: Array, output: *mut Array) -> i32 {
    unsafe { basis::convert(value, Basis::Legendre, true, output) }
}

/// Converts power-basis coefficients to Legendre coefficients.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_poly2leg(value: Array, output: *mut Array) -> i32 {
    unsafe { basis::convert(value, Basis::Legendre, false, output) }
}

binary!(add, terlan_ndarray_v1_legadd, addition);
binary!(subtract, terlan_ndarray_v1_legsub, subtraction);

/// Multiplies two Legendre coefficient vectors.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legmul(
    lhs: Array,
    rhs: Array,
    output: *mut Array,
) -> i32 {
    unsafe { basis::multiply(lhs, rhs, Basis::Legendre, output) }
}

/// Divides two Legendre coefficient vectors into quotient and remainder.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legdiv(
    lhs: Array,
    rhs: Array,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    unsafe { basis::divide(lhs, rhs, Basis::Legendre, output, output_length) }
}

/// Multiplies Legendre coefficients by `x`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legmulx(value: Array, output: *mut Array) -> i32 {
    unsafe { basis::multiply_x(value, Basis::Legendre, output) }
}

/// Raises Legendre coefficients to an integer power.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legpow(
    value: Array,
    exponent: i64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::power(value, exponent, Basis::Legendre, output) }
}

/// Differentiates Legendre coefficients.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legder(
    value: Array,
    order: i64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::derivative(value, order, Basis::Legendre, output) }
}

/// Integrates Legendre coefficients.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legint(
    value: Array,
    order: i64,
    constant: f64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::integral(value, order, constant, Basis::Legendre, output) }
}

/// Fits Legendre coefficients to coordinate/value samples.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legfit(
    coordinates: Array,
    values: Array,
    degree: i64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::fit(coordinates, values, degree, Basis::Legendre, output) }
}

/// Builds Legendre coefficients from roots.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legfromroots(roots: Array, output: *mut Array) -> i32 {
    unsafe { basis::from_roots(roots, Basis::Legendre, output) }
}

unary!(companion, terlan_ndarray_v1_legcompanion, companion);
unary!(roots, terlan_ndarray_v1_legroots, roots);

/// Removes insignificant high-order Legendre coefficients.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legtrim(
    value: Array,
    tolerance: f64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::trim(value, tolerance, output) }
}

/// Returns Legendre coefficients for `offset + scale * x`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legline(
    offset: f64,
    scale: f64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::line(offset, scale, output) }
}

/// Evaluates a Legendre coefficient vector.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legval(
    coordinates: Array,
    coefficients: Array,
    output: *mut Array,
) -> i32 {
    unsafe { basis::evaluate(&[coordinates], coefficients, Basis::Legendre, false, output) }
}

macro_rules! evaluate_nd {
    ($symbol:ident, $grid:expr, $($coordinate:ident),+) => {
        #[doc = "Evaluates a multidimensional Legendre coefficient tensor."]
        #[no_mangle]
        pub unsafe extern "C" fn $symbol(
            $($coordinate: Array,)+
            coefficients: Array,
            output: *mut Array,
        ) -> i32 {
            unsafe { basis::evaluate(&[$($coordinate),+], coefficients, Basis::Legendre, $grid, output) }
        }
    };
}

evaluate_nd!(terlan_ndarray_v1_legval2d, false, x, y);
evaluate_nd!(terlan_ndarray_v1_legval3d, false, x, y, z);
evaluate_nd!(terlan_ndarray_v1_leggrid2d, true, x, y);
evaluate_nd!(terlan_ndarray_v1_leggrid3d, true, x, y, z);

/// Builds a one-dimensional Legendre Vandermonde matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legvander(
    x: Array,
    degree: i64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::vander(&[x], &[degree], Basis::Legendre, output) }
}

/// Builds a two-dimensional Legendre Vandermonde matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legvander2d(
    x: Array,
    y: Array,
    x_degree: i64,
    y_degree: i64,
    output: *mut Array,
) -> i32 {
    unsafe { basis::vander(&[x, y], &[x_degree, y_degree], Basis::Legendre, output) }
}

/// Builds a three-dimensional Legendre Vandermonde matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legvander3d(
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
            Basis::Legendre,
            output,
        )
    }
}

/// Returns Legendre Gaussian quadrature nodes and weights.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_leggauss(degree: i64, output: *mut Array) -> i32 {
    unsafe { basis::gauss(degree, Basis::Legendre, output) }
}

/// Evaluates the Legendre weight function.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_legweight(
    coordinates: Array,
    output: *mut Array,
) -> i32 {
    unsafe { basis::weight(coordinates, Basis::Legendre, output) }
}

#[cfg(test)]
#[path = "ndarray_polynomial_legendre_tests.rs"]
mod tests;
