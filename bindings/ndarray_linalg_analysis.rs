//! NumPy-compatible numerical rank and least-squares analysis.

use faer::{c64, Side};

use super::linalg_support::{hermitian_triangle, linalg_dtypes, MatrixStack};
use super::native_support::{
    delete_arrays, numeric, owned_int_as, owned_numeric_as, owned_real, promoted_dtype, shape,
    transfer_arrays, write_result, Array, DOMAIN_ERROR, INVALID_ARGUMENT, OK, SHAPE_MISMATCH,
};

/// Returns the epsilon associated with one NumPy real-result dtype.
fn dtype_epsilon(dtype: i32) -> f64 {
    if dtype == 11 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    }
}

/// Validates one optional nonnegative absolute tolerance.
fn absolute_tolerance(value: f64, scale: f64, size: usize, dtype: i32) -> Result<f64, i32> {
    if !value.is_finite() || value < 0.0 {
        return Err(DOMAIN_ERROR);
    }
    if value == 0.0 {
        Ok(scale * size as f64 * dtype_epsilon(dtype))
    } else {
        Ok(value)
    }
}

/// Computes numerical ranks for a scalar, vector, matrix, or matrix stack.
fn rank_values(array: Array, tolerance: f64, hermitian: bool) -> Result<(Vec<i64>, Vec<i64>), i32> {
    let dimensions = unsafe { shape(array)? };
    if dimensions.len() < 2 {
        let source = unsafe { numeric(array, true)? };
        let (_, real_dtype) = linalg_dtypes(source.dtype)?;
        let maximum = source
            .values
            .iter()
            .map(|value| value.norm())
            .fold(0.0f64, f64::max);
        let threshold =
            absolute_tolerance(tolerance, maximum, source.values.len().max(1), real_dtype)?;
        return Ok((vec![i64::from(maximum > threshold)], Vec::new()));
    }
    let input = unsafe { MatrixStack::read(array)? };
    if hermitian && input.rows != input.columns {
        return Err(SHAPE_MISMATCH);
    }
    let mut ranks = Vec::with_capacity(input.batches);
    for batch in 0..input.batches {
        let matrix = input.matrix(batch);
        let singular = if input.rows == 0 || input.columns == 0 {
            Vec::new()
        } else if hermitian {
            let selected = hermitian_triangle(&matrix, Side::Lower);
            selected
                .self_adjoint_eigen(Side::Lower)
                .map_err(|_| DOMAIN_ERROR)?
                .S()
                .column_vector()
                .iter()
                .map(|value| value.re.abs())
                .collect::<Vec<_>>()
        } else {
            matrix.singular_values().map_err(|_| DOMAIN_ERROR)?
        };
        let maximum = singular.iter().copied().fold(0.0f64, f64::max);
        let threshold = absolute_tolerance(
            tolerance,
            maximum,
            input.rows.max(input.columns),
            input.real_dtype,
        )?;
        ranks.push(
            i64::try_from(singular.iter().filter(|&&value| value > threshold).count())
                .map_err(|_| SHAPE_MISMATCH)?,
        );
    }
    Ok((ranks, input.batch_shape().to_vec()))
}

/// Computes stack-valued NumPy numerical rank results.
pub(crate) unsafe fn matrix_rank(
    array: Array,
    tolerance: f64,
    hermitian: bool,
    output: *mut Array,
) -> i32 {
    let (values, dimensions) = match rank_values(array, tolerance, hermitian) {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe { write_result(owned_int_as(&values, &dimensions, 7), output) }
}

/// Computes one scalar rank for the legacy rank-two ABI.
pub(crate) unsafe fn matrix_rank_scalar(array: Array, tolerance: f64, output: *mut i64) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let dimensions = match unsafe { shape(array) } {
        Ok(value) if value.len() == 2 => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let _ = dimensions;
    match rank_values(array, tolerance, false) {
        Ok((values, _)) => {
            unsafe { *output = values[0] };
            OK
        }
        Err(status) => status,
    }
}

/// Canonical values produced by one least-squares solve.
struct LeastSquaresResult {
    /// Row-major solution values.
    solution: Vec<c64>,
    /// Solution shape.
    solution_shape: Vec<i64>,
    /// Squared residual norms.
    residuals: Vec<f64>,
    /// Numerical rank.
    rank: i64,
    /// Singular values in descending order.
    singular: Vec<f64>,
    /// Promoted solution dtype.
    value_dtype: i32,
    /// Real residual and singular-value dtype.
    real_dtype: i32,
}

/// Computes the NumPy `lstsq` output model without allocating ABI handles.
unsafe fn least_squares_values(
    coefficients: Array,
    rhs: Array,
    rcond: f64,
) -> Result<LeastSquaresResult, i32> {
    if !rcond.is_finite() {
        return Err(DOMAIN_ERROR);
    }
    let coefficients = unsafe { MatrixStack::read(coefficients)? };
    if coefficients.shape.len() != 2 {
        return Err(SHAPE_MISMATCH);
    }
    let rhs_shape = unsafe { shape(rhs)? };
    if !matches!(rhs_shape.len(), 1 | 2) {
        return Err(SHAPE_MISMATCH);
    }
    let rhs_rows = usize::try_from(rhs_shape[0]).map_err(|_| SHAPE_MISMATCH)?;
    if rhs_rows != coefficients.rows {
        return Err(SHAPE_MISMATCH);
    }
    let rhs_columns = if rhs_shape.len() == 1 {
        1usize
    } else {
        usize::try_from(rhs_shape[1]).map_err(|_| SHAPE_MISMATCH)?
    };
    let rhs_values = unsafe { numeric(rhs, true)? };
    let expected_rhs = coefficients
        .rows
        .checked_mul(rhs_columns)
        .ok_or(SHAPE_MISMATCH)?;
    if rhs_values.values.len() != expected_rhs {
        return Err(SHAPE_MISMATCH);
    }
    let promoted = unsafe { promoted_dtype(coefficients.source_dtype, rhs_values.dtype)? };
    let (value_dtype, real_dtype) = linalg_dtypes(promoted)?;
    let rows = coefficients.rows;
    let columns = coefficients.columns;
    let inner = rows.min(columns);
    let mut solution = vec![c64::new(0.0, 0.0); columns.saturating_mul(rhs_columns)];
    let singular;
    let rank;
    if rows == 0 || columns == 0 {
        singular = Vec::new();
        rank = 0usize;
    } else {
        let matrix = coefficients.matrix(0);
        let decomposition = matrix.thin_svd().map_err(|_| DOMAIN_ERROR)?;
        singular = decomposition
            .S()
            .column_vector()
            .iter()
            .map(|value| value.re)
            .collect::<Vec<_>>();
        let cutoff_ratio = if rcond <= 0.0 {
            dtype_epsilon(real_dtype) * rows.max(columns) as f64
        } else {
            rcond
        };
        let threshold = singular.first().copied().unwrap_or(0.0) * cutoff_ratio;
        rank = singular.iter().filter(|&&value| value > threshold).count();
        for column in 0..columns {
            for rhs_column in 0..rhs_columns {
                let mut value = c64::new(0.0, 0.0);
                for component in 0..inner {
                    if singular[component] > threshold {
                        let projection = (0..rows)
                            .map(|row| {
                                decomposition.U()[(row, component)].conj()
                                    * rhs_values.values[row * rhs_columns + rhs_column]
                            })
                            .sum::<c64>();
                        value += decomposition.V()[(column, component)] * projection
                            / singular[component];
                    }
                }
                solution[column * rhs_columns + rhs_column] = value;
            }
        }
    }
    let residuals = if rank == columns && rows > columns {
        (0..rhs_columns)
            .map(|rhs_column| {
                (0..rows)
                    .map(|row| {
                        let predicted = (0..columns)
                            .map(|column| {
                                coefficients.values[row * columns + column]
                                    * solution[column * rhs_columns + rhs_column]
                            })
                            .sum::<c64>();
                        (predicted - rhs_values.values[row * rhs_columns + rhs_column]).norm_sqr()
                    })
                    .sum::<f64>()
            })
            .collect()
    } else {
        Vec::new()
    };
    let solution_shape = if rhs_shape.len() == 1 {
        vec![columns as i64]
    } else {
        vec![columns as i64, rhs_columns as i64]
    };
    Ok(LeastSquaresResult {
        solution,
        solution_shape,
        residuals,
        rank: i64::try_from(rank).map_err(|_| SHAPE_MISMATCH)?,
        singular,
        value_dtype,
        real_dtype,
    })
}

/// Allocates all four NumPy `lstsq` outputs and transfers them atomically.
pub(crate) unsafe fn least_squares(
    coefficients: Array,
    rhs: Array,
    rcond: f64,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let result = match unsafe { least_squares_values(coefficients, rhs, rcond) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let solution = match unsafe {
        owned_numeric_as(&result.solution, &result.solution_shape, result.value_dtype)
    } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let residual_shape = [result.residuals.len() as i64];
    let residuals =
        match unsafe { owned_real(&result.residuals, &residual_shape, result.real_dtype) } {
            Ok(value) => value,
            Err(status) => {
                unsafe { delete_arrays(&[solution]) };
                return status;
            }
        };
    let rank = match unsafe { owned_int_as(&[result.rank], &[], 7) } {
        Ok(value) => value,
        Err(status) => {
            unsafe { delete_arrays(&[solution, residuals]) };
            return status;
        }
    };
    let singular_shape = [result.singular.len() as i64];
    let singular = match unsafe { owned_real(&result.singular, &singular_shape, result.real_dtype) }
    {
        Ok(value) => value,
        Err(status) => {
            unsafe { delete_arrays(&[solution, residuals, rank]) };
            return status;
        }
    };
    unsafe {
        transfer_arrays(
            vec![solution, residuals, rank, singular],
            output,
            output_length,
        )
    }
}

/// Computes only the solution for the legacy least-squares ABI.
pub(crate) unsafe fn least_squares_solution(
    coefficients: Array,
    rhs: Array,
    tolerance: f64,
    output: *mut Array,
) -> i32 {
    let result = match unsafe { least_squares_values(coefficients, rhs, tolerance) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    unsafe {
        write_result(
            owned_numeric_as(&result.solution, &result.solution_shape, result.value_dtype),
            output,
        )
    }
}

/// Exposes stack-valued numerical ranks through the generated package ABI.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_matrix_rank_general(
    array: Array,
    tolerance: f64,
    hermitian: bool,
    output: *mut Array,
) -> i32 {
    unsafe { matrix_rank(array, tolerance, hermitian, output) }
}

/// Exposes the four NumPy least-squares outputs through the generated package ABI.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_lstsq(
    coefficients: Array,
    rhs: Array,
    rcond: f64,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    unsafe { least_squares(coefficients, rhs, rcond, output, output_length) }
}

#[cfg(test)]
#[path = "ndarray_linalg_analysis_tests.rs"]
mod tests;
