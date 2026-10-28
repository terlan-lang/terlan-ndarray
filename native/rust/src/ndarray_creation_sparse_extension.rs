//! NumPy-compatible sparse coordinate-grid creation.

use rustfft::num_complex::Complex64;

use super::native_support::{
    delete_arrays, owned_numeric_as, transfer_arrays, Array, INVALID_ARGUMENT, OK, SHAPE_MISMATCH,
    SIZE_OVERFLOW,
};

/// Builds one sparse coordinate array for each input dimension.
unsafe fn sparse_indices(shape: &[i64], dtype: i32) -> Result<Vec<Array>, i32> {
    if dtype == 1 || !(2..=14).contains(&dtype) {
        return Err(INVALID_ARGUMENT);
    }
    if shape.iter().any(|dimension| *dimension < 0) {
        return Err(SHAPE_MISMATCH);
    }
    let mut arrays = Vec::with_capacity(shape.len());
    for (axis, &dimension) in shape.iter().enumerate() {
        let length = usize::try_from(dimension).map_err(|_| SIZE_OVERFLOW)?;
        let values = (0..length)
            .map(|coordinate| Complex64::new(coordinate as f64, 0.0))
            .collect::<Vec<_>>();
        let mut sparse_shape = vec![1_i64; shape.len()];
        sparse_shape[axis] = dimension;
        match unsafe { owned_numeric_as(&values, &sparse_shape, dtype) } {
            Ok(array) => arrays.push(array),
            Err(status) => {
                unsafe { delete_arrays(&arrays) };
                return Err(status);
            }
        }
    }
    Ok(arrays)
}

/// Returns sparse coordinate grids with singleton dimensions on every other axis.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_indices_sparse(
    shape: *const i64,
    rank: i64,
    dtype: i32,
    output: *mut *mut u64,
    output_length: *mut usize,
) -> i32 {
    if output.is_null() || output_length.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe {
        *output = std::ptr::null_mut();
        *output_length = 0;
    }
    let rank = match usize::try_from(rank) {
        Ok(rank) => rank,
        Err(_) => return INVALID_ARGUMENT,
    };
    if rank > 0 && shape.is_null() {
        return INVALID_ARGUMENT;
    }
    let shape = if rank == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(shape, rank) }
    };
    match unsafe { sparse_indices(shape, dtype) } {
        Ok(arrays) if arrays.is_empty() => OK,
        Ok(arrays) => unsafe { transfer_arrays(arrays, output, output_length) },
        Err(status) => status,
    }
}
