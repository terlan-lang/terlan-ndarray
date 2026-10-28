//! Shared checked matrix-stack representation for NumPy-compatible linear algebra.

use faer::{c64, Mat, Side};

use super::native_support::{
    element_count, numeric, shape, Array, DOMAIN_ERROR, SHAPE_MISMATCH, UNSUPPORTED_DTYPE,
};

/// Canonical source values and NumPy linear-algebra result dtype policy.
pub(crate) struct MatrixStack {
    /// Complete logical input shape.
    pub(crate) shape: Vec<i64>,
    /// Number of rows in each matrix.
    pub(crate) rows: usize,
    /// Number of columns in each matrix.
    pub(crate) columns: usize,
    /// Number of matrices represented by the leading dimensions.
    pub(crate) batches: usize,
    /// Canonical row-major complex values.
    pub(crate) values: Vec<c64>,
    /// Source ABI dtype code.
    pub(crate) source_dtype: i32,
    /// Dtype used by value-shaped decomposition outputs.
    pub(crate) value_dtype: i32,
    /// Dtype used by real-valued decomposition outputs.
    pub(crate) real_dtype: i32,
}

impl MatrixStack {
    /// Reads one finite rank-two-or-greater array as a matrix stack.
    pub(crate) unsafe fn read(array: Array) -> Result<Self, i32> {
        let shape = unsafe { shape(array)? };
        if shape.len() < 2 {
            return Err(SHAPE_MISMATCH);
        }
        let rows = usize::try_from(shape[shape.len() - 2]).map_err(|_| SHAPE_MISMATCH)?;
        let columns = usize::try_from(shape[shape.len() - 1]).map_err(|_| SHAPE_MISMATCH)?;
        let batches = element_count(&shape[..shape.len() - 2])?;
        let source = unsafe { numeric(array, true)? };
        let (value_dtype, real_dtype) = linalg_dtypes(source.dtype)?;
        let expected = batches
            .checked_mul(rows)
            .and_then(|count| count.checked_mul(columns))
            .ok_or(SHAPE_MISMATCH)?;
        if source.values.len() != expected {
            return Err(SHAPE_MISMATCH);
        }
        Ok(Self {
            shape,
            rows,
            columns,
            batches,
            values: source.values,
            source_dtype: source.dtype,
            value_dtype,
            real_dtype,
        })
    }

    /// Returns the leading batch dimensions.
    pub(crate) fn batch_shape(&self) -> &[i64] {
        &self.shape[..self.shape.len() - 2]
    }

    /// Materializes one batch matrix at canonical computation precision.
    pub(crate) fn matrix(&self, batch: usize) -> Mat<c64> {
        let length = self.rows * self.columns;
        let offset = batch * length;
        Mat::from_fn(self.rows, self.columns, |row, column| {
            self.values[offset + row * self.columns + column]
        })
    }

    /// Builds a result shape by retaining the leading batch dimensions.
    pub(crate) fn result_shape(&self, trailing: &[i64]) -> Vec<i64> {
        self.batch_shape().iter().chain(trailing).copied().collect()
    }
}

/// Returns NumPy's promoted value and real-result dtypes for linear algebra.
pub(crate) fn linalg_dtypes(dtype: i32) -> Result<(i32, i32), i32> {
    match dtype {
        1..=10 => Ok((3, 3)),
        11 => Ok((11, 11)),
        12 => Ok((12, 11)),
        13 => Ok((13, 3)),
        14 => Err(UNSUPPORTED_DTYPE),
        _ => Err(UNSUPPORTED_DTYPE),
    }
}

/// Copies a matrix into row-major canonical values.
pub(crate) fn matrix_values(matrix: &Mat<c64>) -> Vec<c64> {
    let mut values = Vec::with_capacity(matrix.nrows().saturating_mul(matrix.ncols()));
    for row in 0..matrix.nrows() {
        for column in 0..matrix.ncols() {
            values.push(matrix[(row, column)]);
        }
    }
    values
}

/// Appends a row-major identity matrix to one output buffer.
pub(crate) fn append_identity(values: &mut Vec<c64>, rows: usize, columns: usize) {
    for row in 0..rows {
        for column in 0..columns {
            values.push(if row == column {
                c64::new(1.0, 0.0)
            } else {
                c64::new(0.0, 0.0)
            });
        }
    }
}

/// Builds a Hermitian matrix from one selected triangle using NumPy `UPLO` semantics.
pub(crate) fn hermitian_triangle(matrix: &Mat<c64>, side: Side) -> Mat<c64> {
    let dimension = matrix.nrows();
    Mat::from_fn(dimension, dimension, |row, column| {
        if row == column {
            c64::new(matrix[(row, column)].re, 0.0)
        } else {
            match side {
                Side::Lower if row > column => matrix[(row, column)],
                Side::Lower => matrix[(column, row)].conj(),
                Side::Upper if row < column => matrix[(row, column)],
                Side::Upper => matrix[(column, row)].conj(),
            }
        }
    })
}

/// Returns whether one matrix is numerically singular under NumPy's default rank tolerance.
pub(crate) fn is_singular(matrix: &Mat<c64>) -> Result<bool, i32> {
    if matrix.nrows() != matrix.ncols() {
        return Err(SHAPE_MISMATCH);
    }
    if matrix.nrows() == 0 {
        return Ok(false);
    }
    let singular = matrix.singular_values().map_err(|_| DOMAIN_ERROR)?;
    let maximum = singular.first().copied().unwrap_or(0.0);
    let minimum = singular.last().copied().unwrap_or(0.0);
    let threshold = f64::EPSILON * matrix.nrows() as f64 * maximum;
    Ok(minimum <= threshold)
}
