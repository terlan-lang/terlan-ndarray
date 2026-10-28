//! Shared checked access to package-owned native arrays.

use rustfft::num_complex::Complex64;

pub(crate) const OK: i32 = 0;
pub(crate) const INVALID_ARGUMENT: i32 = 100;
pub(crate) const SIZE_OVERFLOW: i32 = 103;
pub(crate) const ALLOCATION_FAILED: i32 = 104;
pub(crate) const DTYPE_MISMATCH: i32 = 106;
pub(crate) const SHAPE_MISMATCH: i32 = 108;
pub(crate) const UNSUPPORTED_DTYPE: i32 = 109;
pub(crate) const AXIS_OUT_OF_RANGE: i32 = 110;
pub(crate) const DOMAIN_ERROR: i32 = 126;

pub(crate) type Array = *mut crate::ffi::TerlanNdArray;

unsafe extern "C" {
    /// Returns the package's canonical promotion result for two ABI dtype codes.
    fn dtype_promote(lhs: i32, rhs: i32) -> i32;
}

/// One logical numeric materialization and its source precision policy.
pub(crate) struct NumericValues {
    /// Logical values represented at Complex128 computation precision.
    pub(crate) values: Vec<Complex64>,
    /// Source ABI dtype code.
    pub(crate) dtype: i32,
    /// Whether the source stores complex scalars.
    pub(crate) complex: bool,
}

/// Reads one array's logical shape.
pub(crate) unsafe fn shape(array: Array) -> Result<Vec<i64>, i32> {
    if array.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    let mut rank = 0i64;
    let status = unsafe { crate::ffi::terlan_ndarray_v1_rank(array, &mut rank) };
    if status != OK {
        return Err(status);
    }
    let rank = usize::try_from(rank).map_err(|_| SHAPE_MISMATCH)?;
    let mut dimensions = std::ptr::null_mut();
    let status = unsafe { crate::ffi::terlan_ndarray_v1_shape(array, &mut dimensions) };
    if status != OK {
        return Err(status);
    }
    if rank == 0 {
        return Ok(Vec::new());
    }
    if dimensions.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    Ok(unsafe { std::slice::from_raw_parts(dimensions, rank) }.to_vec())
}

/// Reads one array's scalar dtype identifier.
pub(crate) unsafe fn dtype(array: Array) -> Result<i32, i32> {
    if array.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    let mut value = 0;
    let status = unsafe { crate::ffi::terlan_ndarray_v1_dtype(array, &mut value) };
    if status == OK {
        Ok(value)
    } else {
        Err(status)
    }
}

/// Copies one array's logical values as `f64`, optionally requiring finiteness.
pub(crate) unsafe fn floats(array: Array, finite: bool) -> Result<Vec<f64>, i32> {
    let mut values = std::ptr::null_mut();
    let mut length = 0usize;
    let status =
        unsafe { crate::ffi::terlan_ndarray_v1_to_floats_cast(array, &mut values, &mut length) };
    if status != OK {
        return Err(status);
    }
    let output = if length == 0 {
        Vec::new()
    } else if values.is_null() {
        return Err(INVALID_ARGUMENT);
    } else {
        unsafe { std::slice::from_raw_parts(values, length) }.to_vec()
    };
    unsafe { crate::ffi::terlan_ndarray_v1_delete_float_values(values) };
    if finite && output.iter().any(|value| !value.is_finite()) {
        Err(DOMAIN_ERROR)
    } else {
        Ok(output)
    }
}

/// Copies one array's logical values through the checked Int64 cast path.
pub(crate) unsafe fn ints(array: Array) -> Result<Vec<i64>, i32> {
    let mut values = std::ptr::null_mut();
    let mut length = 0usize;
    let status =
        unsafe { crate::ffi::terlan_ndarray_v1_to_ints_cast(array, &mut values, &mut length) };
    if status != OK {
        return Err(status);
    }
    let output = if length == 0 {
        Vec::new()
    } else if values.is_null() {
        return Err(INVALID_ARGUMENT);
    } else {
        unsafe { std::slice::from_raw_parts(values, length) }.to_vec()
    };
    unsafe { crate::ffi::terlan_ndarray_v1_delete_int_values(values) };
    Ok(output)
}

/// Takes ownership of one generated array-list result and releases its handle buffer.
#[cfg(test)]
pub(crate) unsafe fn take_arrays(values: *mut u64, length: usize) -> Vec<Array> {
    let arrays = if length == 0 {
        Vec::new()
    } else {
        assert!(!values.is_null());
        unsafe { std::slice::from_raw_parts(values, length) }
            .iter()
            .map(|value| *value as usize as Array)
            .collect()
    };
    unsafe { crate::ffi::terlan_ndarray_v1_delete_handle_values(values) };
    arrays
}

/// Copies one array's logical values through the checked UInt64 cast path.
pub(crate) unsafe fn uints(array: Array) -> Result<Vec<u64>, i32> {
    let mut values = std::ptr::null_mut();
    let mut length = 0usize;
    let status =
        unsafe { crate::ffi::terlan_ndarray_v1_to_uints_cast(array, &mut values, &mut length) };
    if status != OK {
        return Err(status);
    }
    let output = if length == 0 {
        Vec::new()
    } else if values.is_null() {
        return Err(INVALID_ARGUMENT);
    } else {
        unsafe { std::slice::from_raw_parts(values, length) }.to_vec()
    };
    unsafe { crate::ffi::terlan_ndarray_v1_delete_uint_values(values) };
    Ok(output)
}

/// Copies one real or complex array into canonical complex computation values.
pub(crate) unsafe fn numeric(array: Array, finite: bool) -> Result<NumericValues, i32> {
    let dtype = unsafe { dtype(array)? };
    if !matches!(dtype, 1..=14) {
        return Err(UNSUPPORTED_DTYPE);
    }
    if !matches!(dtype, 12 | 13) {
        let values = unsafe { floats(array, finite)? }
            .into_iter()
            .map(|value| Complex64::new(value, 0.0))
            .collect();
        return Ok(NumericValues {
            values,
            dtype,
            complex: false,
        });
    }
    let mut reals = std::ptr::null_mut();
    let mut imags = std::ptr::null_mut();
    let mut real_length = 0usize;
    let mut imag_length = 0usize;
    let real_status =
        unsafe { crate::ffi::terlan_ndarray_v1_complex_reals(array, &mut reals, &mut real_length) };
    let imag_status = if real_status == OK {
        unsafe { crate::ffi::terlan_ndarray_v1_complex_imags(array, &mut imags, &mut imag_length) }
    } else {
        real_status
    };
    if real_status != OK || imag_status != OK || real_length != imag_length {
        unsafe {
            crate::ffi::terlan_ndarray_v1_delete_float_values(reals);
            crate::ffi::terlan_ndarray_v1_delete_float_values(imags);
        }
        return Err(if real_status != OK {
            real_status
        } else {
            imag_status
        });
    }
    let real_values = if real_length == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(reals, real_length) }
    };
    let imag_values = if imag_length == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(imags, imag_length) }
    };
    let values: Vec<Complex64> = real_values
        .iter()
        .zip(imag_values)
        .map(|(&real, &imag)| Complex64::new(real, imag))
        .collect();
    unsafe {
        crate::ffi::terlan_ndarray_v1_delete_float_values(reals);
        crate::ffi::terlan_ndarray_v1_delete_float_values(imags);
    }
    if finite
        && values
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        Err(DOMAIN_ERROR)
    } else {
        Ok(NumericValues {
            values,
            dtype,
            complex: true,
        })
    }
}

/// Computes a checked element count for a nonnegative shape.
pub(crate) fn element_count(shape: &[i64]) -> Result<usize, i32> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        let dimension = usize::try_from(dimension).map_err(|_| SHAPE_MISMATCH)?;
        count.checked_mul(dimension).ok_or(SIZE_OVERFLOW)
    })
}

/// Normalizes one possibly-negative axis against a concrete rank.
pub(crate) fn normalize_axis(axis: i64, rank: usize) -> Result<usize, i32> {
    let rank = i64::try_from(rank).map_err(|_| AXIS_OUT_OF_RANGE)?;
    let axis = if axis < 0 { axis + rank } else { axis };
    if axis < 0 || axis >= rank {
        Err(AXIS_OUT_OF_RANGE)
    } else {
        usize::try_from(axis).map_err(|_| AXIS_OUT_OF_RANGE)
    }
}

/// Constructs one owned real array with an explicit dtype.
pub(crate) unsafe fn owned_real(values: &[f64], shape: &[i64], dtype: i32) -> Result<Array, i32> {
    let mut output = std::ptr::null_mut();
    let length = i64::try_from(values.len()).map_err(|_| SIZE_OVERFLOW)?;
    let rank = i64::try_from(shape.len()).map_err(|_| SIZE_OVERFLOW)?;
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_from_floats_as(
            values.as_ptr(),
            length,
            shape.as_ptr(),
            rank,
            dtype,
            &mut output,
        )
    };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

/// Constructs one owned Bool array from canonical byte values.
pub(crate) unsafe fn owned_bool(values: &[u8], shape: &[i64]) -> Result<Array, i32> {
    let mut output = std::ptr::null_mut();
    let length = i64::try_from(values.len()).map_err(|_| SIZE_OVERFLOW)?;
    let rank = i64::try_from(shape.len()).map_err(|_| SIZE_OVERFLOW)?;
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_from_bools(
            values.as_ptr(),
            length,
            shape.as_ptr(),
            rank,
            &mut output,
        )
    };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

/// Constructs one owned Int64 array from canonical integer values.
pub(crate) unsafe fn owned_int(values: &[i64], shape: &[i64]) -> Result<Array, i32> {
    let mut output = std::ptr::null_mut();
    let length = i64::try_from(values.len()).map_err(|_| SIZE_OVERFLOW)?;
    let rank = i64::try_from(shape.len()).map_err(|_| SIZE_OVERFLOW)?;
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_from_ints(
            values.as_ptr(),
            length,
            shape.as_ptr(),
            rank,
            &mut output,
        )
    };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

/// Constructs one owned integer array with an explicit dtype.
pub(crate) unsafe fn owned_int_as(values: &[i64], shape: &[i64], dtype: i32) -> Result<Array, i32> {
    let mut output = std::ptr::null_mut();
    let length = i64::try_from(values.len()).map_err(|_| SIZE_OVERFLOW)?;
    let rank = i64::try_from(shape.len()).map_err(|_| SIZE_OVERFLOW)?;
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_from_ints_as(
            values.as_ptr(),
            length,
            shape.as_ptr(),
            rank,
            dtype,
            &mut output,
        )
    };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

/// Constructs one owned UInt64 array without a signed or floating conversion.
pub(crate) unsafe fn owned_uint(values: &[u64], shape: &[i64]) -> Result<Array, i32> {
    let mut output = std::ptr::null_mut();
    let length = i64::try_from(values.len()).map_err(|_| SIZE_OVERFLOW)?;
    let rank = i64::try_from(shape.len()).map_err(|_| SIZE_OVERFLOW)?;
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_from_uints(
            values.as_ptr(),
            length,
            shape.as_ptr(),
            rank,
            &mut output,
        )
    };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

/// Constructs one owned complex array with an explicit dtype.
pub(crate) unsafe fn owned_complex(
    values: &[Complex64],
    shape: &[i64],
    dtype: i32,
) -> Result<Array, i32> {
    let reals: Vec<f64> = values.iter().map(|value| value.re).collect();
    let imags: Vec<f64> = values.iter().map(|value| value.im).collect();
    let mut output = std::ptr::null_mut();
    let length = i64::try_from(values.len()).map_err(|_| SIZE_OVERFLOW)?;
    let rank = i64::try_from(shape.len()).map_err(|_| SIZE_OVERFLOW)?;
    let status = unsafe {
        crate::ffi::terlan_ndarray_v1_from_complex(
            reals.as_ptr(),
            length,
            imags.as_ptr(),
            length,
            shape.as_ptr(),
            rank,
            dtype,
            &mut output,
        )
    };
    if status == OK {
        Ok(output)
    } else {
        Err(status)
    }
}

/// Constructs a real or complex result under a source-derived precision policy.
pub(crate) unsafe fn owned_numeric(
    values: &[Complex64],
    shape: &[i64],
    complex: bool,
    narrow: bool,
) -> Result<Array, i32> {
    if complex {
        unsafe { owned_complex(values, shape, if narrow { 12 } else { 13 }) }
    } else {
        let reals: Vec<f64> = values.iter().map(|value| value.re).collect();
        unsafe { owned_real(&reals, shape, if narrow { 11 } else { 3 }) }
    }
}

/// Constructs one numeric result using an explicit canonical ABI dtype.
pub(crate) unsafe fn owned_numeric_as(
    values: &[Complex64],
    shape: &[i64],
    dtype: i32,
) -> Result<Array, i32> {
    match dtype {
        1 => {
            let values = values
                .iter()
                .map(|value| u8::from(value.re != 0.0 || value.im != 0.0))
                .collect::<Vec<_>>();
            unsafe { owned_bool(&values, shape) }
        }
        2 | 4..=10 => {
            let values = values
                .iter()
                .map(|value| {
                    if value.im != 0.0
                        || !value.re.is_finite()
                        || value.re.fract() != 0.0
                        || value.re < i64::MIN as f64
                        || value.re > i64::MAX as f64
                    {
                        Err(DOMAIN_ERROR)
                    } else {
                        Ok(value.re as i64)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            unsafe { owned_int_as(&values, shape, dtype) }
        }
        3 | 11 | 14 => {
            if values.iter().any(|value| value.im != 0.0) {
                return Err(DOMAIN_ERROR);
            }
            let values = values.iter().map(|value| value.re).collect::<Vec<_>>();
            unsafe { owned_real(&values, shape, dtype) }
        }
        12 | 13 => unsafe { owned_complex(values, shape, dtype) },
        _ => Err(UNSUPPORTED_DTYPE),
    }
}

/// Returns the shared C dtype-promotion result without duplicating its table.
pub(crate) unsafe fn promoted_dtype(lhs: i32, rhs: i32) -> Result<i32, i32> {
    let dtype = unsafe { dtype_promote(lhs, rhs) };
    if dtype == 0 {
        Err(UNSUPPORTED_DTYPE)
    } else {
        Ok(dtype)
    }
}

/// Writes a successful owned array or returns its construction status.
pub(crate) unsafe fn write_result(result: Result<Array, i32>, output: *mut Array) -> i32 {
    if output.is_null() {
        if let Ok(array) = result {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(array) };
        }
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    match result {
        Ok(array) => {
            unsafe { *output = array };
            OK
        }
        Err(status) => status,
    }
}

/// Atomically transfers a nonempty list of owned arrays through the generated list ABI.
pub(crate) unsafe fn transfer_arrays(
    arrays: Vec<Array>,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    if output.is_null() || output_length.is_null() || arrays.is_empty() {
        unsafe { delete_arrays(&arrays) };
        return INVALID_ARGUMENT;
    }
    unsafe {
        *output = std::ptr::null_mut();
        *output_length = 0;
    }
    let bytes = match arrays.len().checked_mul(std::mem::size_of::<u64>()) {
        Some(bytes) => bytes,
        None => {
            unsafe { delete_arrays(&arrays) };
            return SIZE_OVERFLOW;
        }
    };
    let identities = unsafe { libc::malloc(bytes) }.cast::<u64>();
    if identities.is_null() {
        unsafe { delete_arrays(&arrays) };
        return ALLOCATION_FAILED;
    }
    for (index, &array) in arrays.iter().enumerate() {
        if array.is_null() {
            unsafe { libc::free(identities.cast()) };
            unsafe { delete_arrays(&arrays) };
            return INVALID_ARGUMENT;
        }
        unsafe { *identities.add(index) = array as usize as u64 };
    }
    unsafe {
        *output = identities;
        *output_length = arrays.len();
    }
    OK
}

/// Deletes every non-null package-owned array in a slice.
pub(crate) unsafe fn delete_arrays(arrays: &[Array]) {
    for &array in arrays {
        if !array.is_null() {
            let _ = unsafe { crate::ffi::terlan_ndarray_v1_delete(array) };
        }
    }
}
