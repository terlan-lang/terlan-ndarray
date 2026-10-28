//! NumPy-compatible stacked matrix systems and general eigen operations.

use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{c64, Mat};

use super::linalg_support::{is_singular, linalg_dtypes, matrix_values, MatrixStack};
use super::native_support::{
    delete_arrays, element_count, numeric, owned_numeric_as, owned_real, promoted_dtype, shape,
    transfer_arrays, write_result, Array, DOMAIN_ERROR, INVALID_ARGUMENT, OK, SHAPE_MISMATCH,
};
use super::polynomial_power_extension::{broadcast_offset, broadcast_shape, strides};

/// Requires square matrices in one checked stack.
fn require_square(input: &MatrixStack) -> Result<usize, i32> {
    if input.rows == input.columns {
        Ok(input.rows)
    } else {
        Err(SHAPE_MISMATCH)
    }
}

/// Computes determinants for every matrix in one stack.
fn determinant_values(input: &MatrixStack) -> Result<Vec<c64>, i32> {
    let dimension = require_square(input)?;
    let mut values = Vec::with_capacity(input.batches);
    for batch in 0..input.batches {
        values.push(if dimension == 0 {
            c64::new(1.0, 0.0)
        } else {
            input.matrix(batch).determinant()
        });
    }
    Ok(values)
}

/// Computes a scalar determinant for the legacy rank-two value ABI.
pub(crate) unsafe fn determinant_scalar(array: Array, output: *mut f64) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) if value.shape.len() == 2 => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let values = match determinant_values(&input) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let value = values[0];
    if value.im != 0.0 {
        return DOMAIN_ERROR;
    }
    unsafe { *output = value.re };
    OK
}

/// Computes one determinant per matrix and preserves leading batch dimensions.
pub(crate) unsafe fn determinant(array: Array, output: *mut Array) -> i32 {
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match determinant_values(&input) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe {
        write_result(
            owned_numeric_as(&values, input.batch_shape(), input.value_dtype),
            output,
        )
    }
}

/// Computes the inverse of every square matrix in a stack.
pub(crate) unsafe fn inverse(array: Array, output: *mut Array) -> i32 {
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let dimension = match require_square(&input) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut values = Vec::with_capacity(input.values.len());
    for batch in 0..input.batches {
        if dimension == 0 {
            continue;
        }
        let matrix = input.matrix(batch);
        match is_singular(&matrix) {
            Ok(false) => values.extend(matrix_values(&matrix.partial_piv_lu().inverse())),
            Ok(true) => return DOMAIN_ERROR,
            Err(status) => return status,
        }
    }
    unsafe {
        write_result(
            owned_numeric_as(&values, &input.shape, input.value_dtype),
            output,
        )
    }
}

/// Canonical right-hand-side layout for a broadcast matrix solve.
struct RightHandSide {
    /// Leading source batch dimensions.
    batch_shape: Vec<i64>,
    /// Canonical row-major source values.
    values: Vec<c64>,
    /// Number of independent right-hand-side columns.
    columns: usize,
    /// Whether the source uses NumPy's vector signature.
    vector: bool,
    /// ABI dtype code.
    dtype: i32,
}

impl RightHandSide {
    /// Reads and validates a solve right-hand side against one coefficient stack.
    unsafe fn read(array: Array, coefficients: &MatrixStack) -> Result<Self, i32> {
        let shape = unsafe { shape(array)? };
        let vector = shape.len() + 1 == coefficients.shape.len();
        let (batch_shape, rows, columns) = if vector {
            if shape.is_empty() {
                return Err(SHAPE_MISMATCH);
            }
            (
                shape[..shape.len() - 1].to_vec(),
                shape[shape.len() - 1],
                1usize,
            )
        } else {
            if shape.len() < 2 {
                return Err(SHAPE_MISMATCH);
            }
            (
                shape[..shape.len() - 2].to_vec(),
                shape[shape.len() - 2],
                usize::try_from(shape[shape.len() - 1]).map_err(|_| SHAPE_MISMATCH)?,
            )
        };
        if usize::try_from(rows).map_err(|_| SHAPE_MISMATCH)? != coefficients.rows {
            return Err(SHAPE_MISMATCH);
        }
        let source = unsafe { numeric(array, true)? };
        let batches = element_count(&batch_shape)?;
        let expected = batches
            .checked_mul(coefficients.rows)
            .and_then(|count| count.checked_mul(columns))
            .ok_or(SHAPE_MISMATCH)?;
        if source.values.len() != expected {
            return Err(SHAPE_MISMATCH);
        }
        Ok(Self {
            batch_shape,
            values: source.values,
            columns,
            vector,
            dtype: source.dtype,
        })
    }

    /// Materializes one source batch selected by a broadcast offset.
    fn matrix(&self, batch: usize, rows: usize) -> Mat<c64> {
        let length = rows * self.columns;
        let offset = batch * length;
        Mat::from_fn(rows, self.columns, |row, column| {
            self.values[offset + row * self.columns + column]
        })
    }
}

/// Solves broadcast-compatible stacks of square linear systems.
pub(crate) unsafe fn solve(coefficients: Array, rhs: Array, output: *mut Array) -> i32 {
    let coefficients = match unsafe { MatrixStack::read(coefficients) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let dimension = match require_square(&coefficients) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs = match unsafe { RightHandSide::read(rhs, &coefficients) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let batch_shape = match broadcast_shape(&[coefficients.batch_shape(), &rhs.batch_shape]) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let batches = match element_count(&batch_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let output_strides = match strides(&batch_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let coefficient_strides = match strides(coefficients.batch_shape()) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let rhs_strides = match strides(&rhs.batch_shape) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let promoted = match unsafe { promoted_dtype(coefficients.source_dtype, rhs.dtype) }
        .and_then(|dtype| linalg_dtypes(dtype).map(|pair| pair.0))
    {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut values = Vec::with_capacity(
        batches
            .saturating_mul(dimension)
            .saturating_mul(rhs.columns),
    );
    for batch in 0..batches {
        if dimension == 0 || rhs.columns == 0 {
            continue;
        }
        let coefficient_batch = broadcast_offset(
            batch,
            &batch_shape,
            &output_strides,
            coefficients.batch_shape(),
            &coefficient_strides,
        );
        let rhs_batch = broadcast_offset(
            batch,
            &batch_shape,
            &output_strides,
            &rhs.batch_shape,
            &rhs_strides,
        );
        let matrix = coefficients.matrix(coefficient_batch);
        match is_singular(&matrix) {
            Ok(false) => values.extend(matrix_values(
                &matrix
                    .partial_piv_lu()
                    .solve(&rhs.matrix(rhs_batch, dimension)),
            )),
            Ok(true) => return DOMAIN_ERROR,
            Err(status) => return status,
        }
    }
    let mut output_shape = batch_shape;
    output_shape.push(dimension as i64);
    if !rhs.vector {
        output_shape.push(rhs.columns as i64);
    }
    unsafe { write_result(owned_numeric_as(&values, &output_shape, promoted), output) }
}

/// Computes one admitted matrix norm for condition-number evaluation.
fn condition_norm(matrix: &Mat<c64>, order: i64) -> Result<f64, i32> {
    match order {
        1 | -1 => {
            let mut selected = if order == 1 { 0.0f64 } else { f64::INFINITY };
            for column in 0..matrix.ncols() {
                let sum = (0..matrix.nrows())
                    .map(|row| matrix[(row, column)].norm())
                    .sum::<f64>();
                selected = if order == 1 {
                    selected.max(sum)
                } else {
                    selected.min(sum)
                };
            }
            Ok(selected)
        }
        3 | -3 => {
            let mut selected = if order == 3 { 0.0f64 } else { f64::INFINITY };
            for row in 0..matrix.nrows() {
                let sum = (0..matrix.ncols())
                    .map(|column| matrix[(row, column)].norm())
                    .sum::<f64>();
                selected = if order == 3 {
                    selected.max(sum)
                } else {
                    selected.min(sum)
                };
            }
            Ok(selected)
        }
        4 => Ok(matrix
            .col_iter()
            .flat_map(|column| column.iter().map(|value| value.norm_sqr()))
            .sum::<f64>()
            .sqrt()),
        _ => Err(INVALID_ARGUMENT),
    }
}

/// Computes condition numbers for every matrix in one stack.
fn condition_values(input: &MatrixStack, order: i64) -> Result<Vec<f64>, i32> {
    let mut values = Vec::with_capacity(input.batches);
    for batch in 0..input.batches {
        let matrix = input.matrix(batch);
        if matrix.nrows() == 0 || matrix.ncols() == 0 {
            return Err(SHAPE_MISMATCH);
        }
        let singular = matrix.singular_values().map_err(|_| DOMAIN_ERROR)?;
        let largest = singular.first().copied().unwrap_or(0.0);
        let smallest = singular.last().copied().unwrap_or(0.0);
        let threshold = f64::EPSILON * matrix.nrows().max(matrix.ncols()) as f64 * largest;
        let singular_matrix = smallest <= threshold;
        values.push(match order {
            0 | 2 => {
                if singular_matrix {
                    f64::INFINITY
                } else {
                    largest / smallest
                }
            }
            -2 => {
                if largest == 0.0 {
                    0.0
                } else {
                    smallest / largest
                }
            }
            1 | -1 | 3 | -3 | 4 => {
                require_square(input)?;
                if singular_matrix {
                    if matches!(order, -1 | -3) {
                        0.0
                    } else {
                        f64::INFINITY
                    }
                } else {
                    let inverse = matrix.partial_piv_lu().inverse();
                    condition_norm(&matrix, order)? * condition_norm(&inverse, order)?
                }
            }
            _ => return Err(INVALID_ARGUMENT),
        });
    }
    Ok(values)
}

/// Computes a rank-two condition number for the legacy scalar ABI.
pub(crate) unsafe fn condition_scalar(array: Array, order: i64, output: *mut f64) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) if value.shape.len() == 2 => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    match condition_values(&input, order) {
        Ok(values) => {
            unsafe { *output = values[0] };
            OK
        }
        Err(status) => status,
    }
}

/// Computes one condition number per matrix in a stack.
pub(crate) unsafe fn condition(array: Array, order: i64, output: *mut Array) -> i32 {
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let values = match condition_values(&input, order) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe {
        write_result(
            owned_real(&values, input.batch_shape(), input.real_dtype),
            output,
        )
    }
}

/// Returns the NumPy result dtype for a completed general eigendecomposition.
fn eigen_dtype(input: &MatrixStack, values: &[c64]) -> i32 {
    if matches!(input.source_dtype, 12 | 13) {
        input.value_dtype
    } else if values.iter().any(|value| value.im != 0.0) {
        if input.value_dtype == 11 {
            12
        } else {
            13
        }
    } else {
        input.value_dtype
    }
}

/// Computes general eigenvalues and optionally right eigenvectors for a stack.
fn eigen_values(input: &MatrixStack, vectors: bool) -> Result<(Vec<c64>, Vec<c64>), i32> {
    let dimension = require_square(input)?;
    let mut values = Vec::with_capacity(input.batches.saturating_mul(dimension));
    let mut vector_values = Vec::with_capacity(input.values.len());
    for batch in 0..input.batches {
        if dimension == 0 {
            continue;
        }
        let matrix = input.matrix(batch);
        if vectors {
            let factor = matrix.eigen().map_err(|_| DOMAIN_ERROR)?;
            values.extend(factor.S().column_vector().iter().copied());
            vector_values.extend(matrix_values(&factor.U().to_owned()));
        } else {
            values.extend(matrix.eigenvalues().map_err(|_| DOMAIN_ERROR)?);
        }
    }
    Ok((values, vector_values))
}

/// Computes stacked general eigenvalues and right eigenvectors.
pub(crate) unsafe fn eig(array: Array, output: *mut *mut u64, output_length: *mut usize) -> i32 {
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let (values, vectors) = match eigen_values(&input, true) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let dtype = eigen_dtype(&input, &values);
    let value_shape = input.result_shape(&[input.rows as i64]);
    let values = match unsafe { owned_numeric_as(&values, &value_shape, dtype) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let vectors = match unsafe { owned_numeric_as(&vectors, &input.shape, dtype) } {
        Ok(value) => value,
        Err(status) => {
            unsafe { delete_arrays(&[values]) };
            return status;
        }
    };
    unsafe { transfer_arrays(vec![values, vectors], output, output_length) }
}

/// Computes stacked general eigenvalues without materializing eigenvectors.
pub(crate) unsafe fn eigvals(array: Array, output: *mut Array) -> i32 {
    let input = match unsafe { MatrixStack::read(array) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let (values, _) = match eigen_values(&input, false) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let dtype = eigen_dtype(&input, &values);
    let value_shape = input.result_shape(&[input.rows as i64]);
    unsafe { write_result(owned_numeric_as(&values, &value_shape, dtype), output) }
}

/// Exposes stacked determinant arrays through the generated package ABI.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_det(array: Array, output: *mut Array) -> i32 {
    unsafe { determinant(array, output) }
}

/// Exposes stacked condition-number arrays through the generated package ABI.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_condition(
    array: Array,
    order: i64,
    output: *mut Array,
) -> i32 {
    unsafe { condition(array, order, output) }
}

#[cfg(test)]
#[path = "ndarray_linalg_system_tests.rs"]
mod tests;
