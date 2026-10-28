//! Shared generalized vector and matrix norm kernels.

use faer::{c64, Mat};

use super::native_support::{DOMAIN_ERROR, INVALID_ARGUMENT};

/// One normalized public norm-order request.
#[derive(Clone, Copy)]
pub(super) enum GeneralNormOrder {
    /// Uses Euclidean or Frobenius behavior selected by reduced rank.
    Default,
    /// Selects the minimum absolute vector value or minimum matrix row sum.
    NegativeInfinity,
    /// Selects the maximum absolute vector value or maximum matrix row sum.
    Infinity,
    /// Selects the matrix Frobenius norm.
    Frobenius,
    /// Selects the matrix nuclear norm.
    Nuclear,
    /// Selects one finite numerical order.
    Finite(f64),
}

/// Computes one vector norm over canonical complex values.
pub(super) fn vector_norm(values: &[c64], order: GeneralNormOrder) -> Result<f64, i32> {
    let magnitudes = values.iter().map(|value| value.norm()).collect::<Vec<_>>();
    match order {
        GeneralNormOrder::Default | GeneralNormOrder::Finite(2.0) => Ok(magnitudes
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()),
        GeneralNormOrder::NegativeInfinity => {
            Ok(magnitudes.into_iter().fold(f64::INFINITY, f64::min))
        }
        GeneralNormOrder::Infinity => Ok(magnitudes.into_iter().fold(0.0, f64::max)),
        GeneralNormOrder::Finite(0.0) => {
            Ok(magnitudes.into_iter().filter(|value| *value != 0.0).count() as f64)
        }
        GeneralNormOrder::Finite(order) if order.is_finite() => {
            if order < 0.0 && magnitudes.iter().any(|value| *value == 0.0) {
                return Ok(0.0);
            }
            Ok(magnitudes
                .iter()
                .map(|value| value.powf(order))
                .sum::<f64>()
                .powf(1.0 / order))
        }
        GeneralNormOrder::Frobenius | GeneralNormOrder::Nuclear => Err(INVALID_ARGUMENT),
        GeneralNormOrder::Finite(_) => Err(INVALID_ARGUMENT),
    }
}

/// Computes one matrix norm over a canonical complex matrix.
pub(super) fn matrix_norm(matrix: &Mat<c64>, order: GeneralNormOrder) -> Result<f64, i32> {
    match order {
        GeneralNormOrder::Default | GeneralNormOrder::Frobenius => Ok(matrix
            .col_iter()
            .flat_map(|column| column.iter().map(|value| value.norm_sqr()))
            .sum::<f64>()
            .sqrt()),
        GeneralNormOrder::Infinity | GeneralNormOrder::NegativeInfinity => {
            let maximum = matches!(order, GeneralNormOrder::Infinity);
            let mut selected = if maximum { 0.0f64 } else { f64::INFINITY };
            for row in 0..matrix.nrows() {
                let sum = (0..matrix.ncols())
                    .map(|column| matrix[(row, column)].norm())
                    .sum::<f64>();
                selected = if maximum {
                    selected.max(sum)
                } else {
                    selected.min(sum)
                };
            }
            Ok(selected)
        }
        GeneralNormOrder::Finite(order) if matches!(order, 1.0 | -1.0) => {
            let mut selected = if order > 0.0 { 0.0f64 } else { f64::INFINITY };
            for column in 0..matrix.ncols() {
                let sum = (0..matrix.nrows())
                    .map(|row| matrix[(row, column)].norm())
                    .sum::<f64>();
                selected = if order > 0.0 {
                    selected.max(sum)
                } else {
                    selected.min(sum)
                };
            }
            Ok(selected)
        }
        GeneralNormOrder::Finite(order) if matches!(order, 2.0 | -2.0) => {
            if matrix.nrows() == 0 || matrix.ncols() == 0 {
                return Ok(0.0);
            }
            let singular = matrix.singular_values().map_err(|_| DOMAIN_ERROR)?;
            Ok(if order > 0.0 {
                singular.first().copied().unwrap_or(0.0)
            } else {
                singular.last().copied().unwrap_or(0.0)
            })
        }
        GeneralNormOrder::Nuclear => {
            if matrix.nrows() == 0 || matrix.ncols() == 0 {
                return Ok(0.0);
            }
            Ok(matrix
                .singular_values()
                .map_err(|_| DOMAIN_ERROR)?
                .into_iter()
                .sum())
        }
        GeneralNormOrder::Finite(_) => Err(INVALID_ARGUMENT),
    }
}

/// Decodes one row-major flat index into logical coordinates.
pub(super) fn flat_coordinates(mut index: usize, dimensions: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; dimensions.len()];
    for axis in (0..dimensions.len()).rev() {
        let dimension = dimensions[axis];
        if dimension > 0 {
            coordinates[axis] = index % dimension;
            index /= dimension;
        }
    }
    coordinates
}

/// Projects input coordinates into one reduced output index.
pub(super) fn output_index(
    coordinates: &[usize],
    dimensions: &[usize],
    reduced: &[bool],
    keepdims: bool,
) -> usize {
    let mut index = 0usize;
    for axis in 0..dimensions.len() {
        if reduced[axis] && !keepdims {
            continue;
        }
        let dimension = if reduced[axis] { 1 } else { dimensions[axis] };
        index = index * dimension + if reduced[axis] { 0 } else { coordinates[axis] };
    }
    index
}
