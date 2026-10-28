//! NumPy-compatible stacked matrix decompositions over the maintained `faer` core.

use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::qr::no_pivoting::factor::{qr_in_place, qr_in_place_scratch};
use faer::{c64, Side};

use super::linalg_support::{append_identity, hermitian_triangle, matrix_values, MatrixStack};
use super::native_support::{
    owned_numeric_as, owned_real, transfer_arrays, write_result, Array, DOMAIN_ERROR,
    INVALID_ARGUMENT, SHAPE_MISMATCH,
};

/// Computes stacked lower-triangular Cholesky factors.
pub(crate) unsafe fn cholesky(array: Array, output: *mut Array) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) if value.rows == value.columns => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let mut values = Vec::with_capacity(input.values.len());
    if input.rows > 0 {
        for batch in 0..input.batches {
            let factor = match input.matrix(batch).llt(Side::Lower) {
                Ok(value) => value,
                Err(_) => return DOMAIN_ERROR,
            };
            values.extend(matrix_values(&factor.L().to_owned()));
        }
    }
    unsafe {
        write_result(
            owned_numeric_as(&values, &input.shape, input.value_dtype),
            output,
        )
    }
}

/// Computes stacked reduced or complete QR factors and transfers `[Q, R]`.
pub(crate) unsafe fn qr(
    array: Array,
    reduced: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let inner = input.rows.min(input.columns);
    let q_columns = if reduced { inner } else { input.rows };
    let r_rows = if reduced { inner } else { input.rows };
    let mut q_values = Vec::new();
    let mut r_values = Vec::new();
    for batch in 0..input.batches {
        if input.rows == 0 || input.columns == 0 {
            append_identity(&mut q_values, input.rows, q_columns);
            r_values.resize(
                r_values.len() + r_rows.saturating_mul(input.columns),
                c64::new(0.0, 0.0),
            );
            continue;
        }
        let factor = input.matrix(batch).qr();
        let q = if reduced {
            factor.compute_thin_Q()
        } else {
            factor.compute_Q()
        };
        q_values.extend(matrix_values(&q));
        let compact = factor.thin_R();
        for row in 0..r_rows {
            for column in 0..input.columns {
                r_values.push(if row < inner {
                    compact[(row, column)]
                } else {
                    c64::new(0.0, 0.0)
                });
            }
        }
    }
    let q_shape = input.result_shape(&[input.rows as i64, q_columns as i64]);
    let r_shape = input.result_shape(&[r_rows as i64, input.columns as i64]);
    let q = match unsafe { owned_numeric_as(&q_values, &q_shape, input.value_dtype) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let r = match unsafe { owned_numeric_as(&r_values, &r_shape, input.value_dtype) } {
        Ok(value) => value,
        Err(status) => {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(q) };
            return status;
        }
    };
    unsafe { transfer_arrays(vec![q, r], output, output_length) }
}

/// Selects one NumPy QR result contract.
enum QrMode {
    /// Returns reduced Q and R factors.
    Reduced,
    /// Returns complete Q and R factors.
    Complete,
    /// Returns only the reduced R factor.
    R,
    /// Returns transposed compact reflectors and their coefficients.
    Raw,
}

impl TryFrom<i64> for QrMode {
    type Error = i32;

    /// Decodes the stable package mode code.
    fn try_from(value: i64) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Reduced),
            1 => Ok(Self::Complete),
            2 => Ok(Self::R),
            3 => Ok(Self::Raw),
            _ => Err(INVALID_ARGUMENT),
        }
    }
}

/// Computes only the reduced R factor for every matrix in a stack.
unsafe fn qr_r(array: Array, output: *mut *mut u64, output_length: *mut usize) -> i32 {
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let inner = input.rows.min(input.columns);
    let mut values = Vec::with_capacity(
        input
            .batches
            .saturating_mul(inner)
            .saturating_mul(input.columns),
    );
    for batch in 0..input.batches {
        if input.rows == 0 || input.columns == 0 {
            continue;
        }
        values.extend(matrix_values(&input.matrix(batch).qr().thin_R().to_owned()));
    }
    let dimensions = input.result_shape(&[inner as i64, input.columns as i64]);
    let result = match unsafe { owned_numeric_as(&values, &dimensions, input.value_dtype) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { transfer_arrays(vec![result], output, output_length) }
}

/// Computes NumPy-compatible compact Householder QR outputs through `faer`.
unsafe fn qr_raw(array: Array, output: *mut *mut u64, output_length: *mut usize) -> i32 {
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let inner = input.rows.min(input.columns);
    let mut reflector_values = Vec::with_capacity(input.values.len());
    let mut coefficient_values = Vec::with_capacity(input.batches.saturating_mul(inner));
    let parallelism = faer::get_global_parallelism();
    for batch in 0..input.batches {
        let mut compact = input.matrix(batch);
        let mut coefficients = faer::Mat::<c64>::zeros(1, inner);
        let requirement = qr_in_place_scratch::<c64>(
            input.rows,
            input.columns,
            1,
            parallelism,
            Default::default(),
        );
        let mut memory = MemBuffer::new(requirement);
        qr_in_place(
            compact.as_mut(),
            coefficients.as_mut(),
            parallelism,
            &mut MemStack::new(&mut memory),
            Default::default(),
        );
        for row in 0..input.columns {
            for column in 0..input.rows {
                reflector_values.push(compact[(column, row)]);
            }
        }
        coefficient_values.extend((0..inner).map(|index| coefficients[(0, index)]));
    }
    let reflector_shape = input.result_shape(&[input.columns as i64, input.rows as i64]);
    let coefficient_shape = input.result_shape(&[inner as i64]);
    let reflectors =
        match unsafe { owned_numeric_as(&reflector_values, &reflector_shape, input.value_dtype) } {
            Ok(value) => value,
            Err(status) => return status,
        };
    let coefficients = match unsafe {
        owned_numeric_as(&coefficient_values, &coefficient_shape, input.value_dtype)
    } {
        Ok(value) => value,
        Err(status) => {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(reflectors) };
            return status;
        }
    };
    unsafe { transfer_arrays(vec![reflectors, coefficients], output, output_length) }
}

/// Computes a QR result under one typed NumPy mode contract.
pub(crate) unsafe fn qr_mode(
    array: Array,
    mode: i64,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    match QrMode::try_from(mode) {
        Ok(QrMode::Reduced) => unsafe { qr(array, true, output, output_length) },
        Ok(QrMode::Complete) => unsafe { qr(array, false, output, output_length) },
        Ok(QrMode::R) => unsafe { qr_r(array, output, output_length) },
        Ok(QrMode::Raw) => unsafe { qr_raw(array, output, output_length) },
        Err(status) => status,
    }
}

/// Computes stacked reduced or complete singular-value factors.
pub(crate) unsafe fn svd(
    array: Array,
    full_matrices: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let inner = input.rows.min(input.columns);
    let u_columns = if full_matrices { input.rows } else { inner };
    let vh_rows = if full_matrices { input.columns } else { inner };
    let mut u_values = Vec::new();
    let mut singular_values = Vec::new();
    let mut vh_values = Vec::new();
    for batch in 0..input.batches {
        if input.rows == 0 || input.columns == 0 {
            append_identity(&mut u_values, input.rows, u_columns);
            append_identity(&mut vh_values, vh_rows, input.columns);
            continue;
        }
        let matrix = input.matrix(batch);
        let factor = match if full_matrices {
            matrix.svd()
        } else {
            matrix.thin_svd()
        } {
            Ok(value) => value,
            Err(_) => return DOMAIN_ERROR,
        };
        u_values.extend(matrix_values(&factor.U().to_owned()));
        singular_values.extend(factor.S().column_vector().iter().map(|value| value.re));
        for row in 0..factor.V().ncols() {
            for column in 0..factor.V().nrows() {
                vh_values.push(factor.V()[(column, row)].conj());
            }
        }
    }
    let u_shape = input.result_shape(&[input.rows as i64, u_columns as i64]);
    let singular_shape = input.result_shape(&[inner as i64]);
    let vh_shape = input.result_shape(&[vh_rows as i64, input.columns as i64]);
    let u = match unsafe { owned_numeric_as(&u_values, &u_shape, input.value_dtype) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let singular = match unsafe { owned_real(&singular_values, &singular_shape, input.real_dtype) }
    {
        Ok(value) => value,
        Err(status) => {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(u) };
            return status;
        }
    };
    let vh = match unsafe { owned_numeric_as(&vh_values, &vh_shape, input.value_dtype) } {
        Ok(value) => value,
        Err(status) => {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(u) };
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(singular) };
            return status;
        }
    };
    unsafe { transfer_arrays(vec![u, singular, vh], output, output_length) }
}

/// Computes stacked singular values without materializing singular vectors.
pub(crate) unsafe fn singular_values(array: Array, output: *mut Array) -> i32 {
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let inner = input.rows.min(input.columns);
    let mut values = Vec::with_capacity(input.batches.saturating_mul(inner));
    for batch in 0..input.batches {
        if inner == 0 {
            continue;
        }
        let singular = match input.matrix(batch).singular_values() {
            Ok(value) => value,
            Err(_) => return DOMAIN_ERROR,
        };
        values.extend(singular);
    }
    let dimensions = input.result_shape(&[inner as i64]);
    unsafe { write_result(owned_real(&values, &dimensions, input.real_dtype), output) }
}

/// Computes stacked Hermitian eigenvalues and optionally eigenvectors.
unsafe fn hermitian_eigen(
    array: Array,
    side: Side,
    vectors: bool,
) -> Result<(MatrixStack, Vec<f64>, Vec<c64>), i32> {
    let input = unsafe { MatrixStack::read(array)? };
    if input.rows != input.columns {
        return Err(SHAPE_MISMATCH);
    }
    let mut eigenvalues = Vec::with_capacity(input.batches.saturating_mul(input.rows));
    let mut eigenvectors = Vec::new();
    for batch in 0..input.batches {
        if input.rows == 0 {
            continue;
        }
        let matrix = hermitian_triangle(&input.matrix(batch), side);
        let factor = matrix
            .self_adjoint_eigen(Side::Lower)
            .map_err(|_| DOMAIN_ERROR)?;
        eigenvalues.extend(factor.S().column_vector().iter().map(|value| value.re));
        if vectors {
            eigenvectors.extend(matrix_values(&factor.U().to_owned()));
        }
    }
    Ok((input, eigenvalues, eigenvectors))
}

/// Computes stacked Hermitian eigenvalues and eigenvectors.
pub(crate) unsafe fn symmetric_eigen(
    array: Array,
    upper: bool,
    tolerance: f64,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return DOMAIN_ERROR;
    }
    let side = if upper { Side::Upper } else { Side::Lower };
    let (input, eigenvalues, eigenvectors) = match unsafe { hermitian_eigen(array, side, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values_shape = input.result_shape(&[input.rows as i64]);
    let vectors_shape = input.shape.clone();
    let values = match unsafe { owned_real(&eigenvalues, &values_shape, input.real_dtype) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let vectors =
        match unsafe { owned_numeric_as(&eigenvectors, &vectors_shape, input.value_dtype) } {
            Ok(value) => value,
            Err(status) => {
                let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(values) };
                return status;
            }
        };
    unsafe { transfer_arrays(vec![values, vectors], output, output_length) }
}

/// Computes stacked Hermitian eigenvalues without materializing eigenvectors.
pub(crate) unsafe fn eigvalsh(array: Array, upper: bool, output: *mut Array) -> i32 {
    let side = if upper { Side::Upper } else { Side::Lower };
    let (input, eigenvalues, _) = match unsafe { hermitian_eigen(array, side, false) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values_shape = input.result_shape(&[input.rows as i64]);
    unsafe {
        write_result(
            owned_real(&eigenvalues, &values_shape, input.real_dtype),
            output,
        )
    }
}

/// Exposes selected-triangle Hermitian eigenvectors through the generated package ABI.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_symmetric_eigen_general(
    array: Array,
    tolerance: f64,
    upper: bool,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    unsafe { symmetric_eigen(array, upper, tolerance, output, output_length) }
}

/// Exposes selected-triangle Hermitian eigenvalues through the generated package ABI.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_eigvalsh_general(
    array: Array,
    upper: bool,
    output: *mut Array,
) -> i32 {
    unsafe { eigvalsh(array, upper, output) }
}

/// Exposes typed NumPy QR modes through the generated package ABI.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_qr_mode(
    array: Array,
    mode: i64,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    unsafe { qr_mode(array, mode, output, output_length) }
}

/// Exposes values-only singular-value decomposition through the generated package ABI.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_singular_values(
    array: Array,
    output: *mut Array,
) -> i32 {
    unsafe { singular_values(array, output) }
}

#[cfg(test)]
#[path = "ndarray_linalg_decomposition_tests.rs"]
mod tests;
