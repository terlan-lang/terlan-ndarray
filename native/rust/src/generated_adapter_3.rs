use super::*;

impl Array {
    /// Returns a truncated Int from an array containing exactly one real element.
    pub fn item_int(&self) -> Result<i64, CAbiError> {
        let mut out_out_value: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_item_int(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.array.item_int", status, 0)?;
        Ok(out_out_value)
    }

    /// Returns one Bool element after normalizing a complete index tuple.
    pub fn get_bool(&self, indices: &[i64]) -> Result<bool, CAbiError> {
        let indices_length = i64::try_from(indices.len()).map_err(|_| CAbiError { operation: "ndarray.array.get_bool", status: -2 })?;
        let mut out_out_value: bool = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_get_bool(self.raw.as_ptr(), indices.as_ptr(), indices_length, &mut out_out_value) };
        check_status("ndarray.array.get_bool", status, 0)?;
        Ok(out_out_value)
    }

    /// Returns one Int64 element after normalizing a complete index tuple.
    pub fn get_int(&self, indices: &[i64]) -> Result<i64, CAbiError> {
        let indices_length = i64::try_from(indices.len()).map_err(|_| CAbiError { operation: "ndarray.array.get_int", status: -2 })?;
        let mut out_out_value: i64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_get_int(self.raw.as_ptr(), indices.as_ptr(), indices_length, &mut out_out_value) };
        check_status("ndarray.array.get_int", status, 0)?;
        Ok(out_out_value)
    }

    /// Returns one Float64 element after normalizing a complete index tuple.
    pub fn get_float(&self, indices: &[i64]) -> Result<f64, CAbiError> {
        let indices_length = i64::try_from(indices.len()).map_err(|_| CAbiError { operation: "ndarray.array.get_float", status: -2 })?;
        let mut out_out_value: f64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_get_float(self.raw.as_ptr(), indices.as_ptr(), indices_length, &mut out_out_value) };
        check_status("ndarray.array.get_float", status, 0)?;
        Ok(out_out_value)
    }

    /// Assigns one Bool element through a writable view and makes the mutation visible to its aliases.
    pub fn set_bool(&mut self, indices: &[i64], value: bool) -> Result<(), CAbiError> {
        let indices_length = i64::try_from(indices.len()).map_err(|_| CAbiError { operation: "ndarray.array.set_bool", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_set_bool(self.raw.as_ptr(), indices.as_ptr(), indices_length, value) };
        check_status("ndarray.array.set_bool", status, 0)?;
        Ok(())
    }

    /// Assigns one Int64 element through a writable view and makes the mutation visible to its aliases.
    pub fn set_int(&mut self, indices: &[i64], value: i64) -> Result<(), CAbiError> {
        let indices_length = i64::try_from(indices.len()).map_err(|_| CAbiError { operation: "ndarray.array.set_int", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_set_int(self.raw.as_ptr(), indices.as_ptr(), indices_length, value) };
        check_status("ndarray.array.set_int", status, 0)?;
        Ok(())
    }

    /// Assigns one Float64 element through a writable view and makes the mutation visible to its aliases.
    pub fn set_float(&mut self, indices: &[i64], value: f64) -> Result<(), CAbiError> {
        let indices_length = i64::try_from(indices.len()).map_err(|_| CAbiError { operation: "ndarray.array.set_float", status: -2 })?;
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_set_float(self.raw.as_ptr(), indices.as_ptr(), indices_length, value) };
        check_status("ndarray.array.set_float", status, 0)?;
        Ok(())
    }

    /// Adds broadcast-compatible numeric arrays with canonical dtype promotion.
    pub fn add(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_add(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.add", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.add", status: -1 })?;
        Ok(Self { raw })
    }

    /// Subtracts broadcast-compatible numeric arrays with canonical dtype promotion.
    pub fn subtract(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_subtract(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.subtract", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.subtract", status: -1 })?;
        Ok(Self { raw })
    }

    /// Multiplies broadcast-compatible numeric arrays with canonical dtype promotion.
    pub fn multiply(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_multiply(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.multiply", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.multiply", status: -1 })?;
        Ok(Self { raw })
    }

    /// Performs vector, matrix, and broadcast-batched matrix multiplication with numeric dtype promotion.
    pub fn matmul(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_matmul(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.matmul", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.matmul", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes NumPy-compatible scalar multiplication, vector inner products, and generalized array dot products.
    pub fn dot(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_dot(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.dot", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.dot", status: -1 })?;
        Ok(Self { raw })
    }

    /// Contracts the final dimensions of two arrays without conjugation.
    pub fn inner(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_inner(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.inner", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.inner", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes the outer product of the flattened inputs.
    pub fn outer(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_outer(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.outer", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.outer", status: -1 })?;
        Ok(Self { raw })
    }

    /// Flattens both inputs and computes their conjugating vector dot product.
    pub fn vdot(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_vdot(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.vdot", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.vdot", status: -1 })?;
        Ok(Self { raw })
    }

    /// Sums a selected diagonal over two distinct axes.
    pub fn trace(&self, offset: i64, axis_a: i64, axis_b: i64) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_trace(self.raw.as_ptr(), offset, axis_a, axis_b, &mut raw) };
        check_status("ndarray.array.trace", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.trace", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes a finite real square matrix determinant through maintained faer factorization.
    pub fn determinant(&self) -> Result<f64, CAbiError> {
        let mut out_out_value: f64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_determinant(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.array.determinant", status, 0)?;
        Ok(out_out_value)
    }

    /// Solves a finite real square system for a vector or matrix right-hand side through maintained faer factorization.
    pub fn solve(&self, rhs: &Self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_solve(self.raw.as_ptr(), rhs.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.solve", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.solve", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes a finite real square matrix inverse by solving against identity through maintained faer factorization.
    pub fn inverse(&self) -> Result<Self, CAbiError> {
        let mut raw: *mut ffi::TerlanNdArray = std::ptr::null_mut();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_inverse(self.raw.as_ptr(), &mut raw) };
        check_status("ndarray.array.inverse", status, 0)?;
        let raw = NonNull::new(raw).ok_or(CAbiError { operation: "ndarray.array.inverse", status: -1 })?;
        Ok(Self { raw })
    }

    /// Computes a vector one-norm or matrix induced one-norm.
    pub fn norm_l1(&self) -> Result<f64, CAbiError> {
        let mut out_out_value: f64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_norm_l1(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.array.norm_l1", status, 0)?;
        Ok(out_out_value)
    }

    /// Computes a vector Euclidean norm or matrix spectral norm.
    pub fn norm_l2(&self) -> Result<f64, CAbiError> {
        let mut out_out_value: f64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_norm_l2(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.array.norm_l2", status, 0)?;
        Ok(out_out_value)
    }

    /// Computes a vector maximum norm or matrix induced infinity norm.
    pub fn norm_max(&self) -> Result<f64, CAbiError> {
        let mut out_out_value: f64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_norm_max(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.array.norm_max", status, 0)?;
        Ok(out_out_value)
    }

    /// Computes a vector Euclidean norm or matrix Frobenius norm.
    pub fn frobenius_norm(&self) -> Result<f64, CAbiError> {
        let mut out_out_value: f64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_frobenius_norm(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.array.frobenius_norm", status, 0)?;
        Ok(out_out_value)
    }

    /// Computes the nuclear norm of a finite real matrix.
    pub fn nuclear_norm(&self) -> Result<f64, CAbiError> {
        let mut out_out_value: f64 = Default::default();
        // SAFETY: generated arguments follow the reviewed ownership metadata.
        let status = unsafe { ffi::terlan_ndarray_v1_nuclear_norm(self.raw.as_ptr(), &mut out_out_value) };
        check_status("ndarray.array.nuclear_norm", status, 0)?;
        Ok(out_out_value)
    }

}
