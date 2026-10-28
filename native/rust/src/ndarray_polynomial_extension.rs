//! Polynomial algebra backed by the package's maintained `faer` kernels.

use faer::Mat;
use rustfft::num_complex::Complex64;

use super::native_support::{
    numeric, owned_complex, owned_numeric, shape, transfer_arrays, write_result, Array,
    NumericValues, DOMAIN_ERROR, INVALID_ARGUMENT, SHAPE_MISMATCH, SIZE_OVERFLOW,
    UNSUPPORTED_DTYPE,
};

/// Reads one nonempty finite rank-one numeric vector without losing complex values.
pub(crate) unsafe fn numeric_vector(array: Array) -> Result<NumericValues, i32> {
    let dimensions = unsafe { shape(array)? };
    if dimensions.len() != 1 || dimensions[0] <= 0 {
        return Err(SHAPE_MISMATCH);
    }
    unsafe { numeric(array, true) }
}

/// Reads one rank-one numeric vector while admitting an empty coefficient sequence.
unsafe fn numeric_vector_or_empty(array: Array) -> Result<NumericValues, i32> {
    let dimensions = unsafe { shape(array)? };
    if dimensions.len() != 1 {
        return Err(SHAPE_MISMATCH);
    }
    unsafe { numeric(array, true) }
}

/// Returns whether one source dtype uses the narrow floating precision policy.
pub(crate) fn narrow_dtype(dtype: i32) -> bool {
    matches!(dtype, 11 | 12 | 14)
}

/// Evaluates low-to-high complex power coefficients at one coordinate.
fn evaluate_numeric(coefficients: &[Complex64], coordinate: Complex64) -> Complex64 {
    coefficients
        .iter()
        .rev()
        .fold(Complex64::new(0.0, 0.0), |value, coefficient| {
            value * coordinate + coefficient
        })
}

/// Removes redundant high-order complex zeros while preserving one constant.
pub(crate) fn trim_numeric(coefficients: &[Complex64]) -> &[Complex64] {
    let last = coefficients
        .iter()
        .rposition(|coefficient| *coefficient != Complex64::new(0.0, 0.0))
        .unwrap_or(0);
    &coefficients[..=last]
}

/// Adds or subtracts aligned low-to-high complex coefficients.
fn aligned_numeric(lhs: &[Complex64], rhs: &[Complex64], subtract: bool) -> Vec<Complex64> {
    let length = lhs.len().max(rhs.len());
    let mut output = vec![Complex64::new(0.0, 0.0); length];
    for (index, value) in lhs.iter().enumerate() {
        output[index] += value;
    }
    for (index, value) in rhs.iter().enumerate() {
        output[index] += if subtract { -*value } else { *value };
    }
    trim_numeric(&output).to_vec()
}

/// Multiplies low-to-high complex coefficients by convolution.
pub(crate) fn multiplied_numeric(
    lhs: &[Complex64],
    rhs: &[Complex64],
) -> Result<Vec<Complex64>, i32> {
    if lhs.is_empty() || rhs.is_empty() {
        return Ok(Vec::new());
    }
    let length = lhs
        .len()
        .checked_add(rhs.len())
        .and_then(|value| value.checked_sub(1))
        .ok_or(SIZE_OVERFLOW)?;
    let mut output = vec![Complex64::new(0.0, 0.0); length];
    for (lhs_index, lhs_value) in lhs.iter().enumerate() {
        for (rhs_index, rhs_value) in rhs.iter().enumerate() {
            output[lhs_index + rhs_index] += lhs_value * rhs_value;
        }
    }
    Ok(trim_numeric(&output).to_vec())
}

/// Divides low-to-high complex coefficients into quotient and remainder.
fn divided_numeric(
    dividend: &[Complex64],
    divisor: &[Complex64],
) -> Result<(Vec<Complex64>, Vec<Complex64>), i32> {
    let dividend = trim_numeric(dividend);
    let divisor = trim_numeric(divisor);
    if divisor.last() == Some(&Complex64::new(0.0, 0.0)) {
        return Err(DOMAIN_ERROR);
    }
    if dividend.len() < divisor.len() {
        return Ok((vec![Complex64::new(0.0, 0.0)], dividend.to_vec()));
    }
    let mut remainder = dividend.to_vec();
    let mut quotient = vec![Complex64::new(0.0, 0.0); dividend.len() - divisor.len() + 1];
    for index in (0..quotient.len()).rev() {
        let remainder_degree = index + divisor.len() - 1;
        let factor = remainder[remainder_degree] / divisor[divisor.len() - 1];
        quotient[index] = factor;
        for (offset, coefficient) in divisor.iter().enumerate() {
            remainder[index + offset] -= factor * coefficient;
        }
    }
    let remainder_length = divisor.len().saturating_sub(1).max(1);
    let remainder = trim_numeric(&remainder[..remainder_length.min(remainder.len())]).to_vec();
    Ok((trim_numeric(&quotient).to_vec(), remainder))
}

/// Solves one finite least-squares matrix with rank diagnostics.
pub(crate) fn least_squares(matrix: &Mat<f64>, rhs: &[f64]) -> Result<Vec<f64>, i32> {
    if matrix.nrows() != rhs.len() || matrix.nrows() < matrix.ncols() || matrix.ncols() == 0 {
        return Err(SHAPE_MISMATCH);
    }
    let decomposition = matrix.thin_svd().map_err(|_| DOMAIN_ERROR)?;
    let singular: Vec<f64> = decomposition.S().column_vector().iter().copied().collect();
    let maximum = singular.first().copied().unwrap_or(0.0);
    let tolerance = f64::EPSILON * matrix.nrows().max(matrix.ncols()) as f64 * maximum;
    if singular.iter().filter(|&&value| value > tolerance).count() < matrix.ncols() {
        return Err(DOMAIN_ERROR);
    }
    let mut output = vec![0.0; matrix.ncols()];
    for column in 0..matrix.ncols() {
        for component in 0..singular.len() {
            let projection = (0..matrix.nrows())
                .map(|row| decomposition.U()[(row, component)] * rhs[row])
                .sum::<f64>();
            output[column] +=
                decomposition.V()[(column, component)] * projection / singular[component];
        }
    }
    Ok(output)
}

/// Returns basis values from degree zero through `degree` at one coordinate.
fn basis_values(basis: i32, degree: usize, coordinate: f64) -> Result<Vec<f64>, i32> {
    if !matches!(basis, 0..=2) {
        return Err(INVALID_ARGUMENT);
    }
    let mut values = vec![0.0; degree + 1];
    values[0] = 1.0;
    if degree == 0 {
        return Ok(values);
    }
    values[1] = coordinate;
    for order in 2..=degree {
        values[order] = match basis {
            0 => values[order - 1] * coordinate,
            1 => 2.0 * coordinate * values[order - 1] - values[order - 2],
            2 => {
                ((2 * order - 1) as f64 * coordinate * values[order - 1]
                    - (order - 1) as f64 * values[order - 2])
                    / order as f64
            }
            _ => unreachable!(),
        };
    }
    Ok(values)
}

/// Evaluates low-to-high polynomial coefficients at every coordinate.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyval(
    coordinates: Array,
    coefficients: Array,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let coefficients = match unsafe { numeric_vector(coefficients) } {
        Ok(values) => values,
        Err(status) => return status,
    };
    let dimensions = match unsafe { shape(coordinates) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let coordinates = match unsafe { numeric(coordinates, true) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let complex = coefficients.complex || coordinates.complex;
    let narrow = narrow_dtype(coefficients.dtype) && narrow_dtype(coordinates.dtype);
    let values: Vec<Complex64> = coordinates
        .values
        .into_iter()
        .map(|coordinate| evaluate_numeric(&coefficients.values, coordinate))
        .collect();
    unsafe { write_result(owned_numeric(&values, &dimensions, complex, narrow), output) }
}

/// Returns all roots of a finite low-to-high polynomial.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyroots(
    coefficients: Array,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let coefficients = match unsafe { numeric_vector(coefficients) } {
        Ok(values) => trim_numeric(&values.values).to_vec(),
        Err(status) => return status,
    };
    if coefficients.len() == 1 {
        return unsafe { write_result(owned_complex(&[], &[0], 13), output) };
    }
    let coefficients = coefficients.into_iter().rev().collect::<Vec<_>>();
    if coefficients[0] == Complex64::new(0.0, 0.0) {
        return DOMAIN_ERROR;
    }
    let degree = coefficients.len() - 1;
    let mut companion = Mat::<Complex64>::zeros(degree, degree);
    for column in 0..degree {
        companion[(0, column)] = -coefficients[column + 1] / coefficients[0];
    }
    for row in 1..degree {
        companion[(row, row - 1)] = Complex64::new(1.0, 0.0);
    }
    let roots = match companion.eigenvalues() {
        Ok(values) => values,
        Err(_) => return DOMAIN_ERROR,
    };
    unsafe { write_result(owned_complex(&roots, &[degree as i64], 13), output) }
}

/// Fits one low-to-high polynomial by finite SVD least squares.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyfit(
    coordinates: Array,
    values: Array,
    degree: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() || degree < 0 {
        return INVALID_ARGUMENT;
    }
    let coordinates = match unsafe { numeric_vector(coordinates) } {
        Ok(value) if !value.complex => value,
        Ok(_) => return UNSUPPORTED_DTYPE,
        Err(status) => return status,
    };
    let values = match unsafe { numeric_vector(values) } {
        Ok(value) if value.values.len() == coordinates.values.len() => value,
        Ok(_) => return SHAPE_MISMATCH,
        Err(status) => return status,
    };
    let columns = match usize::try_from(degree)
        .ok()
        .and_then(|value| value.checked_add(1))
    {
        Some(value) if coordinates.values.len() >= value => value,
        _ => return SHAPE_MISMATCH,
    };
    let vandermonde = Mat::from_fn(coordinates.values.len(), columns, |row, column| {
        coordinates.values[row].re.powi(column as i32)
    });
    let real_values: Vec<f64> = values.values.iter().map(|value| value.re).collect();
    let imaginary_values: Vec<f64> = values.values.iter().map(|value| value.im).collect();
    let real_coefficients = match least_squares(&vandermonde, &real_values) {
        Ok(value) => value,
        Err(status) => return status,
    };
    let imaginary_coefficients = if values.complex {
        match least_squares(&vandermonde, &imaginary_values) {
            Ok(value) => value,
            Err(status) => return status,
        }
    } else {
        vec![0.0; columns]
    };
    let coefficients: Vec<Complex64> = real_coefficients
        .into_iter()
        .zip(imaginary_coefficients)
        .map(|(real, imaginary)| Complex64::new(real, imaginary))
        .collect();
    unsafe {
        write_result(
            owned_numeric(
                &coefficients,
                &[columns as i64],
                values.complex,
                narrow_dtype(values.dtype),
            ),
            output,
        )
    }
}

/// Adds two low-to-high polynomial coefficient vectors.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyadd(
    lhs: Array,
    rhs: Array,
    output: *mut Array,
) -> i32 {
    let result = unsafe { numeric_vector(lhs) }.and_then(|lhs| {
        unsafe { numeric_vector(rhs) }.and_then(|rhs| {
            let values = aligned_numeric(&lhs.values, &rhs.values, false);
            unsafe {
                owned_numeric(
                    &values,
                    &[values.len() as i64],
                    lhs.complex || rhs.complex,
                    narrow_dtype(lhs.dtype) && narrow_dtype(rhs.dtype),
                )
            }
        })
    });
    unsafe { write_result(result, output) }
}

/// Subtracts two low-to-high polynomial coefficient vectors.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polysub(
    lhs: Array,
    rhs: Array,
    output: *mut Array,
) -> i32 {
    let result = unsafe { numeric_vector(lhs) }.and_then(|lhs| {
        unsafe { numeric_vector(rhs) }.and_then(|rhs| {
            let values = aligned_numeric(&lhs.values, &rhs.values, true);
            unsafe {
                owned_numeric(
                    &values,
                    &[values.len() as i64],
                    lhs.complex || rhs.complex,
                    narrow_dtype(lhs.dtype) && narrow_dtype(rhs.dtype),
                )
            }
        })
    });
    unsafe { write_result(result, output) }
}

/// Multiplies two low-to-high polynomial coefficient vectors.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polymul(
    lhs: Array,
    rhs: Array,
    output: *mut Array,
) -> i32 {
    let result = unsafe { numeric_vector_or_empty(lhs) }.and_then(|lhs| {
        unsafe { numeric_vector_or_empty(rhs) }.and_then(|rhs| {
            multiplied_numeric(&lhs.values, &rhs.values).and_then(|values| unsafe {
                owned_numeric(
                    &values,
                    &[values.len() as i64],
                    lhs.complex || rhs.complex,
                    narrow_dtype(lhs.dtype) && narrow_dtype(rhs.dtype),
                )
            })
        })
    });
    unsafe { write_result(result, output) }
}

/// Divides polynomials and atomically returns `[quotient, remainder]`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polydiv_all(
    lhs: Array,
    rhs: Array,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    let (quotient, remainder, complex, narrow) =
        match unsafe { numeric_vector(lhs) }.and_then(|lhs| {
            unsafe { numeric_vector(rhs) }.and_then(|rhs| {
                divided_numeric(&lhs.values, &rhs.values).map(|(quotient, remainder)| {
                    (
                        quotient,
                        remainder,
                        lhs.complex || rhs.complex,
                        narrow_dtype(lhs.dtype) && narrow_dtype(rhs.dtype),
                    )
                })
            })
        }) {
            Ok(values) => values,
            Err(status) => return status,
        };
    let quotient_array =
        match unsafe { owned_numeric(&quotient, &[quotient.len() as i64], complex, narrow) } {
            Ok(value) => value,
            Err(status) => return status,
        };
    let remainder_array =
        match unsafe { owned_numeric(&remainder, &[remainder.len() as i64], complex, narrow) } {
            Ok(value) => value,
            Err(status) => {
                let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(quotient_array) };
                return status;
            }
        };
    unsafe { transfer_arrays(vec![quotient_array, remainder_array], output, output_length) }
}

/// Differentiates a low-to-high polynomial by a nonnegative order.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyder(
    coefficients: Array,
    order: i64,
    output: *mut Array,
) -> i32 {
    if order < 0 {
        return INVALID_ARGUMENT;
    }
    let coefficients = match unsafe { numeric_vector(coefficients) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut values = coefficients.values;
    for _ in 0..order {
        if values.len() <= 1 {
            values = vec![Complex64::new(0.0, 0.0)];
            break;
        }
        values = values
            .into_iter()
            .skip(1)
            .enumerate()
            .map(|(index, value)| value * (index + 1) as f64)
            .collect();
    }
    unsafe {
        write_result(
            owned_numeric(
                &values,
                &[values.len() as i64],
                coefficients.complex,
                narrow_dtype(coefficients.dtype),
            ),
            output,
        )
    }
}

/// Integrates a low-to-high polynomial with one constant per integration.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyint(
    coefficients: Array,
    order: i64,
    constant: f64,
    output: *mut Array,
) -> i32 {
    if order < 0 || !constant.is_finite() {
        return INVALID_ARGUMENT;
    }
    let coefficients = match unsafe { numeric_vector(coefficients) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut values = coefficients.values;
    for _ in 0..order {
        values = std::iter::once(Complex64::new(constant, 0.0))
            .chain(
                values
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| value / (index + 1) as f64),
            )
            .collect();
    }
    unsafe {
        write_result(
            owned_numeric(
                &values,
                &[values.len() as i64],
                coefficients.complex,
                narrow_dtype(coefficients.dtype),
            ),
            output,
        )
    }
}

/// Builds the power-basis companion matrix used by root finding.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polycompanion(
    coefficients: Array,
    output: *mut Array,
) -> i32 {
    let coefficients = match unsafe { numeric_vector(coefficients) } {
        Ok(values) => values,
        Err(status) => return status,
    };
    let values = trim_numeric(&coefficients.values);
    if values.len() < 2 {
        return DOMAIN_ERROR;
    }
    let degree = values.len() - 1;
    let mut companion = vec![Complex64::new(0.0, 0.0); degree * degree];
    let leading = values[degree];
    for row in 0..degree {
        companion[row * degree + degree - 1] = -values[row] / leading;
    }
    for row in 1..degree {
        companion[row * degree + row - 1] = Complex64::new(1.0, 0.0);
    }
    unsafe {
        write_result(
            owned_numeric(
                &companion,
                &[degree as i64, degree as i64],
                coefficients.complex,
                narrow_dtype(coefficients.dtype),
            ),
            output,
        )
    }
}

/// Converts low-to-high coefficients among power, Chebyshev, and Legendre bases.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_polyconvert(
    coefficients: Array,
    source_basis: i32,
    target_basis: i32,
    domain_minimum: f64,
    domain_maximum: f64,
    window_minimum: f64,
    window_maximum: f64,
    output: *mut Array,
) -> i32 {
    if output.is_null()
        || !matches!(source_basis, 0..=2)
        || !matches!(target_basis, 0..=2)
        || ![
            domain_minimum,
            domain_maximum,
            window_minimum,
            window_maximum,
        ]
        .iter()
        .all(|value| value.is_finite())
        || domain_minimum == domain_maximum
        || window_minimum == window_maximum
    {
        return INVALID_ARGUMENT;
    }
    let coefficients = match unsafe { numeric_vector(coefficients) } {
        Ok(values) => values,
        Err(status) => return status,
    };
    let degree = coefficients.values.len() - 1;
    let sample_count = degree + 1;
    let mut targets = vec![Complex64::new(0.0, 0.0); sample_count];
    let mut matrix = Mat::<f64>::zeros(sample_count, sample_count);
    for sample in 0..sample_count {
        let unit = if degree == 0 {
            0.0
        } else {
            ((sample as f64 + 0.5) * std::f64::consts::PI / sample_count as f64).cos()
        };
        let physical = domain_minimum
            + (unit - window_minimum) * (domain_maximum - domain_minimum)
                / (window_maximum - window_minimum);
        let mapped = window_minimum
            + (physical - domain_minimum) * (window_maximum - window_minimum)
                / (domain_maximum - domain_minimum);
        let source_values = match basis_values(source_basis, degree, mapped) {
            Ok(value) => value,
            Err(status) => return status,
        };
        targets[sample] = coefficients
            .values
            .iter()
            .zip(source_values)
            .map(|(coefficient, value)| coefficient * value)
            .sum();
        let target_values = match basis_values(target_basis, degree, mapped) {
            Ok(value) => value,
            Err(status) => return status,
        };
        for column in 0..sample_count {
            matrix[(sample, column)] = target_values[column];
        }
    }
    let real_targets: Vec<f64> = targets.iter().map(|value| value.re).collect();
    let imaginary_targets: Vec<f64> = targets.iter().map(|value| value.im).collect();
    let real_converted = match least_squares(&matrix, &real_targets) {
        Ok(values) => values,
        Err(status) => return status,
    };
    let imaginary_converted = if coefficients.complex {
        match least_squares(&matrix, &imaginary_targets) {
            Ok(values) => values,
            Err(status) => return status,
        }
    } else {
        vec![0.0; sample_count]
    };
    let converted: Vec<Complex64> = real_converted
        .into_iter()
        .zip(imaginary_converted)
        .map(|(real, imaginary)| Complex64::new(real, imaginary))
        .collect();
    unsafe {
        write_result(
            owned_numeric(
                &converted,
                &[sample_count as i64],
                coefficients.complex,
                narrow_dtype(coefficients.dtype),
            ),
            output,
        )
    }
}

#[cfg(test)]
#[path = "ndarray_polynomial_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ndarray_polynomial_numpy_tests.rs"]
mod numpy_tests;
