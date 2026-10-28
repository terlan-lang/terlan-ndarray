//! Maintained `faer` real-valued matrix operations for `terlan-ndarray`.

use faer::Mat;

const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = 100;
const SHAPE_MISMATCH: i32 = 108;
const DOMAIN_ERROR: i32 = 126;

unsafe fn array_shape(array: *mut super::super::ffi::TerlanNdArray) -> Result<Vec<i64>, i32> {
    if array.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    let mut rank = 0i64;
    // SAFETY: the caller provides a live borrowed array and writable rank slot.
    let status = unsafe { super::super::ffi::terlan_ndarray_v1_rank(array, &mut rank) };
    if status != OK {
        return Err(status);
    }
    let rank = usize::try_from(rank).map_err(|_| SHAPE_MISMATCH)?;
    let mut shape = std::ptr::null_mut();
    // SAFETY: the array owns the borrowed shape for this call.
    let status = unsafe { super::super::ffi::terlan_ndarray_v1_shape(array, &mut shape) };
    if status != OK {
        return Err(status);
    }
    if rank == 0 {
        return Ok(Vec::new());
    }
    if shape.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    // SAFETY: successful shape lookup exposes exactly `rank` dimensions.
    Ok(unsafe { std::slice::from_raw_parts(shape, rank) }.to_vec())
}

unsafe fn array_floats(array: *mut super::super::ffi::TerlanNdArray) -> Result<Vec<f64>, i32> {
    let mut values = std::ptr::null_mut();
    let mut length = 0usize;
    // SAFETY: the array is borrowed and the output allocation is package-owned.
    let status = unsafe {
        super::super::ffi::terlan_ndarray_v1_to_floats_cast(array, &mut values, &mut length)
    };
    if status != OK {
        return Err(status);
    }
    let result = if length == 0 {
        Vec::new()
    } else if values.is_null() {
        return Err(INVALID_ARGUMENT);
    } else {
        // SAFETY: successful readback owns `length` initialized doubles.
        unsafe { std::slice::from_raw_parts(values, length) }.to_vec()
    };
    // SAFETY: the matching package destructor accepts null for an empty output.
    unsafe { super::super::ffi::terlan_ndarray_v1_delete_float_values(values) };
    if result.iter().any(|value| !value.is_finite()) {
        return Err(DOMAIN_ERROR);
    }
    Ok(result)
}

fn matrix_from_values(values: &[f64], rows: usize, columns: usize) -> Mat<f64> {
    Mat::from_fn(rows, columns, |row, column| values[row * columns + column])
}

fn singular_values(matrix: &Mat<f64>) -> Result<Vec<f64>, i32> {
    matrix.singular_values().map_err(|_| DOMAIN_ERROR)
}

/// Computes a finite real square matrix determinant through maintained `faer` factorization.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_determinant(
    array: *mut super::super::ffi::TerlanNdArray,
    out_value: *mut f64,
) -> i32 {
    unsafe { super::linalg_system::determinant_scalar(array, out_value) }
}

/// Solves a finite real square system for a vector or matrix right-hand side.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_solve(
    matrix: *mut super::super::ffi::TerlanNdArray,
    rhs: *mut super::super::ffi::TerlanNdArray,
    out_array: *mut *mut super::super::ffi::TerlanNdArray,
) -> i32 {
    unsafe { super::linalg_system::solve(matrix, rhs, out_array) }
}

/// Computes a finite real square matrix inverse by solving against identity.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_inverse(
    matrix: *mut super::super::ffi::TerlanNdArray,
    out_array: *mut *mut super::super::ffi::TerlanNdArray,
) -> i32 {
    unsafe { super::linalg_system::inverse(matrix, out_array) }
}

enum NormOrder {
    One,
    Two,
    Maximum,
    Frobenius,
    Nuclear,
}

unsafe fn norm_value(
    array: *mut super::super::ffi::TerlanNdArray,
    order: NormOrder,
) -> Result<f64, i32> {
    let shape = unsafe { array_shape(array)? };
    if !matches!(shape.len(), 1 | 2) {
        return Err(SHAPE_MISMATCH);
    }
    let values = unsafe { array_floats(array)? };
    if shape.len() == 1 {
        return match order {
            NormOrder::One => Ok(values.iter().map(|value| value.abs()).sum()),
            NormOrder::Two | NormOrder::Frobenius => {
                Ok(values.iter().map(|value| value * value).sum::<f64>().sqrt())
            }
            NormOrder::Maximum => Ok(values.iter().map(|value| value.abs()).fold(0.0, f64::max)),
            NormOrder::Nuclear => Err(SHAPE_MISMATCH),
        };
    }
    let rows = usize::try_from(shape[0]).map_err(|_| SHAPE_MISMATCH)?;
    let columns = usize::try_from(shape[1]).map_err(|_| SHAPE_MISMATCH)?;
    match order {
        NormOrder::One => {
            let mut maximum: f64 = 0.0;
            for column in 0..columns {
                let sum = (0..rows)
                    .map(|row| values[row * columns + column].abs())
                    .sum();
                maximum = maximum.max(sum);
            }
            Ok(maximum)
        }
        NormOrder::Maximum => {
            let mut maximum: f64 = 0.0;
            for row in 0..rows {
                let sum = values[row * columns..(row + 1) * columns]
                    .iter()
                    .map(|value| value.abs())
                    .sum();
                maximum = maximum.max(sum);
            }
            Ok(maximum)
        }
        NormOrder::Frobenius => Ok(values.iter().map(|value| value * value).sum::<f64>().sqrt()),
        NormOrder::Two | NormOrder::Nuclear => {
            if rows == 0 || columns == 0 {
                return Ok(0.0);
            }
            let singular = singular_values(&matrix_from_values(&values, rows, columns))?;
            if matches!(order, NormOrder::Two) {
                Ok(singular.first().copied().unwrap_or(0.0))
            } else {
                Ok(singular.into_iter().sum())
            }
        }
    }
}

unsafe fn write_norm(
    array: *mut super::super::ffi::TerlanNdArray,
    order: NormOrder,
    out_value: *mut f64,
) -> i32 {
    if out_value.is_null() {
        return INVALID_ARGUMENT;
    }
    let value = match unsafe { norm_value(array, order) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *out_value = value };
    OK
}

/// Computes a vector one-norm or matrix induced one-norm.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_norm_l1(
    array: *mut super::super::ffi::TerlanNdArray,
    out_value: *mut f64,
) -> i32 {
    unsafe { write_norm(array, NormOrder::One, out_value) }
}

/// Computes a vector Euclidean norm or matrix spectral norm.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_norm_l2(
    array: *mut super::super::ffi::TerlanNdArray,
    out_value: *mut f64,
) -> i32 {
    unsafe { write_norm(array, NormOrder::Two, out_value) }
}

/// Computes a vector maximum norm or matrix induced infinity norm.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_norm_max(
    array: *mut super::super::ffi::TerlanNdArray,
    out_value: *mut f64,
) -> i32 {
    unsafe { write_norm(array, NormOrder::Maximum, out_value) }
}

/// Computes a vector Euclidean norm or matrix Frobenius norm.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_frobenius_norm(
    array: *mut super::super::ffi::TerlanNdArray,
    out_value: *mut f64,
) -> i32 {
    unsafe { write_norm(array, NormOrder::Frobenius, out_value) }
}

/// Computes the nuclear norm of a finite real matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_nuclear_norm(
    array: *mut super::super::ffi::TerlanNdArray,
    out_value: *mut f64,
) -> i32 {
    unsafe { write_norm(array, NormOrder::Nuclear, out_value) }
}

/// Computes numerical matrix rank from singular values and an explicit tolerance.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_matrix_rank(
    array: *mut super::super::ffi::TerlanNdArray,
    tolerance: f64,
    out_rank: *mut i64,
) -> i32 {
    unsafe { super::linalg_analysis::matrix_rank_scalar(array, tolerance, out_rank) }
}

/// Solves a finite real least-squares system using an SVD pseudoinverse.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_least_squares(
    matrix: *mut super::super::ffi::TerlanNdArray,
    rhs: *mut super::super::ffi::TerlanNdArray,
    tolerance: f64,
    out_array: *mut *mut super::super::ffi::TerlanNdArray,
) -> i32 {
    unsafe { super::linalg_analysis::least_squares_solution(matrix, rhs, tolerance, out_array) }
}

/// Computes a lower-triangular Cholesky factor for a symmetric positive-definite matrix.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_cholesky(
    array: *mut super::super::ffi::TerlanNdArray,
    out_array: *mut *mut super::super::ffi::TerlanNdArray,
) -> i32 {
    unsafe { super::linalg_decomposition::cholesky(array, out_array) }
}

/// Computes reduced or complete QR factors and transfers `[Q, R]`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_qr(
    array: *mut super::super::ffi::TerlanNdArray,
    reduced: i32,
    out_values: *mut *mut u64,
    out_length: *mut usize,
) -> i32 {
    if !matches!(reduced, 0 | 1) {
        return INVALID_ARGUMENT;
    }
    unsafe { super::linalg_decomposition::qr(array, reduced == 1, out_values, out_length) }
}

/// Computes reduced or complete singular-value factors and transfers `[U, S, Vh]`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_svd(
    array: *mut super::super::ffi::TerlanNdArray,
    full_matrices: i32,
    out_values: *mut *mut u64,
    out_length: *mut usize,
) -> i32 {
    if !matches!(full_matrices, 0 | 1) {
        return INVALID_ARGUMENT;
    }
    unsafe { super::linalg_decomposition::svd(array, full_matrices == 1, out_values, out_length) }
}

/// Computes symmetric eigenvalues and eigenvectors and transfers `[values, vectors]`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_symmetric_eigen(
    array: *mut super::super::ffi::TerlanNdArray,
    tolerance: f64,
    out_values: *mut *mut u64,
    out_length: *mut usize,
) -> i32 {
    unsafe {
        super::linalg_decomposition::symmetric_eigen(
            array, false, tolerance, out_values, out_length,
        )
    }
}

#[cfg(test)]
#[path = "ndarray_linear_algebra_tests.rs"]
mod tests;
