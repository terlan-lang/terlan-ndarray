use super::*;

impl Array {
    /// Fills the receiver in place from one scalar array using receiver dtype conversion.
    pub fn fill(&mut self, value: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_fill(self.raw.as_ptr(), value.raw.as_ptr()) };
        check_status("ndarray.array.fill", status, 0)?;
        Ok(())
    }

    /// Returns a resized logical-order array by repeating or truncating source values.
    pub fn resize(&self, shape: &[i64]) -> Result<Self, CAbiError> {
        let shape_length = i64::try_from(shape.len()).map_err(|_| CAbiError { operation: "ndarray.array.resize", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_resize(self.raw.as_ptr(), shape.as_ptr(), shape_length, &mut raw) };
        check_status("ndarray.array.resize", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.resize", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces selected values using NumPy-compatible any semantics and a broadcast boolean mask.
    pub fn any_where(&self, mask: &Self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.any_where", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_any_where(self.raw.as_ptr(), mask.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.any_where", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.any_where", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces selected values using NumPy-compatible all semantics and a broadcast boolean mask.
    pub fn all_where(&self, mask: &Self, axes: &[i64], keep_dims: bool) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.all_where", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_all_where(self.raw.as_ptr(), mask.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.all_where", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.all_where", status: -1 })?;
        Ok(Self { raw })
    }

    /// Selects broadcast choice values and atomically writes them into an exactly shaped destination.
    pub fn choose_into(&mut self, indexes: &Self, choices: &[&Array], mode: i64) -> Result<(), CAbiError> {
        let choices_handles = choices.iter().map(|value| value.raw.as_ptr() as usize as u64).collect::<Vec<_>>();
        let choices_length = i64::try_from(choices.len()).map_err(|_| CAbiError { operation: "ndarray.array.choose_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_choose_into(self.raw.as_ptr(), indexes.raw.as_ptr(), choices_handles.as_ptr(), choices_length, mode as i32) };
        check_status("ndarray.array.choose_into", status, 0)?;
        Ok(())
    }

    /// Rounds every real or complex component to the requested decimal position using ties-to-even semantics.
    pub fn round_decimals(&self, decimals: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_round_decimals(self.raw.as_ptr(), decimals, &mut raw) };
        check_status("ndarray.array.round_decimals", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.round_decimals", status: -1 })?;
        Ok(Self { raw })
    }

    /// Rounds values and atomically writes them into an exactly shaped same-kind destination.
    pub fn round_into(&mut self, array: &Self, decimals: i64) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_round_into(self.raw.as_ptr(), array.raw.as_ptr(), decimals) };
        check_status("ndarray.array.round_into", status, 0)?;
        Ok(())
    }

    /// Conjugates values and atomically writes them into an exactly shaped same-kind destination.
    pub fn conjugate_into(&mut self, array: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_conjugate_into(self.raw.as_ptr(), array.raw.as_ptr()) };
        check_status("ndarray.array.conjugate_into", status, 0)?;
        Ok(())
    }

    /// Multiplies arrays and atomically writes the result into an exactly shaped, exact-dtype destination.
    pub fn matmul_into(&mut self, lhs: &Self, rhs: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_matmul_into(self.raw.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr()) };
        check_status("ndarray.array.matmul_into", status, 0)?;
        Ok(())
    }

    /// Computes a dot product into an exact-shape, exact-dtype, C-contiguous destination.
    pub fn dot_into(&mut self, lhs: &Self, rhs: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_dot_into(self.raw.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr()) };
        check_status("ndarray.array.dot_into", status, 0)?;
        Ok(())
    }

    /// Computes a diagonal sum and atomically writes it into an exactly shaped destination.
    pub fn trace_into(&mut self, array: &Self, offset: i64, axis_a: i64, axis_b: i64) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_trace_into(self.raw.as_ptr(), array.raw.as_ptr(), offset, axis_a, axis_b) };
        check_status("ndarray.array.trace_into", status, 0)?;
        Ok(())
    }

    /// Writes the outer product into an exactly shaped writable destination.
    pub fn outer_into(&mut self, lhs: &Self, rhs: &Self) -> Result<(), CAbiError> {
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_outer_into(self.raw.as_ptr(), lhs.raw.as_ptr(), rhs.raw.as_ptr()) };
        check_status("ndarray.array.outer_into", status, 0)?;
        Ok(())
    }

    /// Sums into an exactly shaped writable destination without broadcasting the reduction result.
    pub fn sum_into(&mut self, array: &Self, axes: &[i64], keep_dims: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.sum_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sum_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.array.sum_into", status, 0)?;
        Ok(())
    }

    /// Multiplies into an exactly shaped writable destination without broadcasting the reduction result.
    pub fn product_into(&mut self, array: &Self, axes: &[i64], keep_dims: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.product_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_product_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.array.product_into", status, 0)?;
        Ok(())
    }

    /// Reduces minima into an exactly shaped writable destination.
    pub fn minimum_into(&mut self, array: &Self, axes: &[i64], keep_dims: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.minimum_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_minimum_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.array.minimum_into", status, 0)?;
        Ok(())
    }

    /// Reduces maxima into an exactly shaped writable destination.
    pub fn maximum_into(&mut self, array: &Self, axes: &[i64], keep_dims: i64) -> Result<(), CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.maximum_into", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_maximum_into(self.raw.as_ptr(), array.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32) };
        check_status("ndarray.array.maximum_into", status, 0)?;
        Ok(())
    }

    /// Sums elements selected by a broadcast-compatible truth mask.
    pub fn sum_where(&self, mask: &Self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.sum_where", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sum_where(self.raw.as_ptr(), mask.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.sum_where", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sum_where", status: -1 })?;
        Ok(Self { raw })
    }

    /// Sums elements using a one-element array as the initial accumulator.
    pub fn sum_initial(&self, initial: &Self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.sum_initial", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sum_initial(self.raw.as_ptr(), initial.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.sum_initial", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sum_initial", status: -1 })?;
        Ok(Self { raw })
    }

    /// Sums selected elements from an explicit one-element initial accumulator.
    pub fn sum_where_initial(&self, mask: &Self, initial: &Self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.sum_where_initial", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_sum_where_initial(self.raw.as_ptr(), mask.raw.as_ptr(), initial.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.sum_where_initial", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.sum_where_initial", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces minima using a one-element initial value, including empty lanes.
    pub fn minimum_initial(&self, initial: &Self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.minimum_initial", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_minimum_initial(self.raw.as_ptr(), initial.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.minimum_initial", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.minimum_initial", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces maxima using a one-element initial value, including empty lanes.
    pub fn maximum_initial(&self, initial: &Self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.maximum_initial", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_maximum_initial(self.raw.as_ptr(), initial.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.maximum_initial", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.maximum_initial", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces minima selected by a mask and rejects selected empty lanes.
    pub fn minimum_where(&self, mask: &Self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.minimum_where", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_minimum_where(self.raw.as_ptr(), mask.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.minimum_where", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.minimum_where", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces maxima selected by a mask and rejects selected empty lanes.
    pub fn maximum_where(&self, mask: &Self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.maximum_where", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_maximum_where(self.raw.as_ptr(), mask.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.maximum_where", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.maximum_where", status: -1 })?;
        Ok(Self { raw })
    }

    /// Reduces selected minima from an explicit one-element initial value.
    pub fn minimum_where_initial(&self, mask: &Self, initial: &Self, axes: &[i64], keep_dims: i64) -> Result<Self, CAbiError> {
        let axes_length = i64::try_from(axes.len()).map_err(|_| CAbiError { operation: "ndarray.array.minimum_where_initial", status: -2 })?;
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_minimum_where_initial(self.raw.as_ptr(), mask.raw.as_ptr(), initial.raw.as_ptr(), axes.as_ptr(), axes_length, keep_dims as i32, &mut raw) };
        check_status("ndarray.array.minimum_where_initial", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.minimum_where_initial", status: -1 })?;
        Ok(Self { raw })
    }

}
