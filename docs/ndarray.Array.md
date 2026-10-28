# ndarray.Array

Generated ABI-v1 bindings for shared strided CPU arrays and owned materializations.

## `from_bools`

Copies boolean values and shape into owned CPU storage.

- C symbol: `terlan_ndarray_v1_from_bools`
- NativeBoundary operation: `ndarray.array.from_bools`
- Ownership: `opaque_handle`

## `from_ints`

Copies Int64 values and shape into owned CPU storage.

- C symbol: `terlan_ndarray_v1_from_ints`
- NativeBoundary operation: `ndarray.array.from_ints`
- Ownership: `opaque_handle`

## `from_floats`

Copies Float64 values and shape into owned CPU storage.

- C symbol: `terlan_ndarray_v1_from_floats`
- NativeBoundary operation: `ndarray.array.from_floats`
- Ownership: `opaque_handle`

## `from_ints_as`

Constructs an array in a checked integer-compatible dtype code.

- C symbol: `terlan_ndarray_v1_from_ints_as`
- NativeBoundary operation: `ndarray.array.from_ints_as`
- Ownership: `opaque_handle`

## `from_floats_as`

Constructs an array in a checked floating-compatible dtype code.

- C symbol: `terlan_ndarray_v1_from_floats_as`
- NativeBoundary operation: `ndarray.array.from_floats_as`
- Ownership: `opaque_handle`

## `from_complex`

Constructs Complex64 or Complex128 storage from parallel real and imaginary values.

- C symbol: `terlan_ndarray_v1_from_complex`
- NativeBoundary operation: `ndarray.array.from_complex`
- Ownership: `opaque_handle`

## `empty`

Allocates an initialized array; native-boundary safety guarantees zeroed storage rather than exposing uninitialized bytes.

- C symbol: `terlan_ndarray_v1_empty`
- NativeBoundary operation: `ndarray.array.empty`
- Ownership: `opaque_handle`

## `zeros`

Creates a zero-filled array in the requested dtype.

- C symbol: `terlan_ndarray_v1_zeros`
- NativeBoundary operation: `ndarray.array.zeros`
- Ownership: `opaque_handle`

## `zeros_ordered`

Creates a zero-filled array with explicit C or Fortran storage order.

- C symbol: `terlan_ndarray_v1_zeros_ordered`
- NativeBoundary operation: `ndarray.array.zeros_ordered`
- Ownership: `opaque_handle`

## `subtract_reduce`

Reduces an axis by ordered subtraction without applying a commutative reassociation.

- C symbol: `terlan_ndarray_v1_subtract_reduce`
- NativeBoundary operation: `ndarray.array.subtract_reduce`
- Ownership: `opaque_handle`

## `subtract_accumulate`

Returns the ordered cumulative subtraction along an axis.

- C symbol: `terlan_ndarray_v1_subtract_accumulate`
- NativeBoundary operation: `ndarray.array.subtract_accumulate`
- Ownership: `opaque_handle`

## `divide_reduce`

Reduces an axis by ordered floating or complex division.

- C symbol: `terlan_ndarray_v1_divide_reduce`
- NativeBoundary operation: `ndarray.array.divide_reduce`
- Ownership: `opaque_handle`

## `divide_accumulate`

Returns the ordered cumulative floating or complex division along an axis.

- C symbol: `terlan_ndarray_v1_divide_accumulate`
- NativeBoundary operation: `ndarray.array.divide_accumulate`
- Ownership: `opaque_handle`

## `ones`

Creates a one-filled array in the requested dtype.

- C symbol: `terlan_ndarray_v1_ones`
- NativeBoundary operation: `ndarray.array.ones`
- Ownership: `opaque_handle`

## `full_int`

Creates an array filled from one checked integer value.

- C symbol: `terlan_ndarray_v1_full_int`
- NativeBoundary operation: `ndarray.array.full_int`
- Ownership: `opaque_handle`

## `full_float`

Creates an array filled from one checked floating value.

- C symbol: `terlan_ndarray_v1_full_float`
- NativeBoundary operation: `ndarray.array.full_float`
- Ownership: `opaque_handle`

## `arange`

Creates consecutive Int64 values from zero to an exclusive integer stop.

- C symbol: `terlan_ndarray_v1_arange_int_stop`
- NativeBoundary operation: `ndarray.array.arange_int_stop`
- Ownership: `opaque_handle`

## `arange_float`

Creates consecutive Float64 values from zero to an exclusive floating stop.

- C symbol: `terlan_ndarray_v1_arange_float_stop`
- NativeBoundary operation: `ndarray.array.arange_float_stop`
- Ownership: `opaque_handle`

## `arange`

Creates exact consecutive Int64 values over a half-open integer interval.

- C symbol: `terlan_ndarray_v1_arange_int_interval`
- NativeBoundary operation: `ndarray.array.arange_int_interval`
- Ownership: `opaque_handle`

## `arange_float`

Creates consecutive Float64 values over a half-open floating interval.

- C symbol: `terlan_ndarray_v1_arange_float_interval`
- NativeBoundary operation: `ndarray.array.arange_float_interval`
- Ownership: `opaque_handle`

## `arange_int`

Creates exact Int64 values over a half-open integer interval.

- C symbol: `terlan_ndarray_v1_arange_int`
- NativeBoundary operation: `ndarray.array.arange_int`
- Ownership: `opaque_handle`

## `arange`

Creates Float64 values over a half-open interval with an explicit step.

- C symbol: `terlan_ndarray_v1_arange_float_step`
- NativeBoundary operation: `ndarray.array.arange_float_step`
- Ownership: `opaque_handle`

## `arange`

Creates evenly stepped values over a half-open interval.

- C symbol: `terlan_ndarray_v1_arange`
- NativeBoundary operation: `ndarray.array.arange`
- Ownership: `opaque_handle`

## `linspace`

Creates a requested number of evenly spaced values.

- C symbol: `terlan_ndarray_v1_linspace`
- NativeBoundary operation: `ndarray.array.linspace`
- Ownership: `opaque_handle`

## `identity`

Creates a rectangular identity-style matrix at an integer diagonal offset.

- C symbol: `terlan_ndarray_v1_identity`
- NativeBoundary operation: `ndarray.array.identity`
- Ownership: `opaque_handle`

## `random_uniform`

Creates reproducible ChaCha8 uniform floating samples in a half-open interval.

- C symbol: `terlan_ndarray_v1_random_uniform`
- NativeBoundary operation: `ndarray.array.random_uniform`
- Ownership: `opaque_handle`

## `random_normal`

Creates reproducible ChaCha8 normal floating samples with explicit distribution parameters.

- C symbol: `terlan_ndarray_v1_random_normal`
- NativeBoundary operation: `ndarray.array.random_normal`
- Ownership: `opaque_handle`

## `random_integers`

Creates reproducible integer samples in a half-open interval.

- C symbol: `terlan_ndarray_v1_random_integers`
- NativeBoundary operation: `ndarray.array.random_integers`
- Ownership: `opaque_handle`

## `random_bernoulli`

Creates reproducible Boolean samples from a Bernoulli distribution.

- C symbol: `terlan_ndarray_v1_random_bernoulli`
- NativeBoundary operation: `ndarray.array.random_bernoulli`
- Ownership: `opaque_handle`

## `random_permutation`

Creates a reproducibly shuffled permutation of consecutive integers.

- C symbol: `terlan_ndarray_v1_random_permutation`
- NativeBoundary operation: `ndarray.array.random_permutation`
- Ownership: `opaque_handle`

## `random_shuffle`

Returns an independently owned array reproducibly shuffled along one axis.

- C symbol: `terlan_ndarray_v1_random_shuffle`
- NativeBoundary operation: `ndarray.array.random_shuffle`
- Ownership: `opaque_handle`

## `diagonal`

Returns a shared strided diagonal view across two distinct axes.

- C symbol: `terlan_ndarray_v1_diagonal`
- NativeBoundary operation: `ndarray.array.diagonal`
- Ownership: `opaque_handle`

## `rank`

Returns array rank.

- C symbol: `terlan_ndarray_v1_rank`
- NativeBoundary operation: `ndarray.array.rank`
- Ownership: `borrowed_handle`

## `shape`

Copies array shape into a Terlan list.

- C symbol: `terlan_ndarray_v1_shape`
- NativeBoundary operation: `ndarray.array.shape`
- Ownership: `value`

## `numel`

Returns logical element count.

- C symbol: `terlan_ndarray_v1_numel`
- NativeBoundary operation: `ndarray.array.numel`
- Ownership: `borrowed_handle`

## `strides`

Copies the array's logical element strides, including zero and negative view strides.

- C symbol: `terlan_ndarray_v1_strides`
- NativeBoundary operation: `ndarray.array.strides`
- Ownership: `value`

## `byte_count`

Returns the checked byte count of the logical array elements.

- C symbol: `terlan_ndarray_v1_byte_count`
- NativeBoundary operation: `ndarray.array.byte_count`
- Ownership: `borrowed_handle`

## `dtype_code`

Returns the ABI-v1 dtype code.

- C symbol: `terlan_ndarray_v1_dtype`
- NativeBoundary operation: `ndarray.array.dtype_code`
- Ownership: `borrowed_handle`

## `contiguous_code`

Returns one when the current view is canonical C-contiguous.

- C symbol: `terlan_ndarray_v1_is_contiguous`
- NativeBoundary operation: `ndarray.array.contiguous_code`
- Ownership: `borrowed_handle`

## `fortran_contiguous_code`

Returns one when the current view is Fortran-contiguous.

- C symbol: `terlan_ndarray_v1_is_fortran_contiguous`
- NativeBoundary operation: `ndarray.array.fortran_contiguous_code`
- Ownership: `borrowed_handle`

## `device_type_code`

Returns the ABI-v1 CPU device code.

- C symbol: `terlan_ndarray_v1_device_type`
- NativeBoundary operation: `ndarray.array.device_type_code`
- Ownership: `borrowed_handle`

## `to_bools`

Returns an owned copy of Bool values.

- C symbol: `terlan_ndarray_v1_to_bools`
- NativeBoundary operation: `ndarray.array.to_bools`
- Ownership: `value`

## `to_ints`

Returns an owned copy of Int64 values.

- C symbol: `terlan_ndarray_v1_to_ints`
- NativeBoundary operation: `ndarray.array.to_ints`
- Ownership: `value`

## `to_floats`

Returns an owned copy of Float64 values.

- C symbol: `terlan_ndarray_v1_to_floats`
- NativeBoundary operation: `ndarray.array.to_floats`
- Ownership: `value`

## `to_ints_cast`

Returns checked integral values from any exactly representable non-complex dtype.

- C symbol: `terlan_ndarray_v1_to_ints_cast`
- NativeBoundary operation: `ndarray.array.to_ints_cast`
- Ownership: `value`

## `to_floats_cast`

Returns real values converted to Terlan Float from any non-complex dtype.

- C symbol: `terlan_ndarray_v1_to_floats_cast`
- NativeBoundary operation: `ndarray.array.to_floats_cast`
- Ownership: `value`

## `complex_reals`

Returns the real components of a complex array.

- C symbol: `terlan_ndarray_v1_complex_reals`
- NativeBoundary operation: `ndarray.array.complex_reals`
- Ownership: `value`

## `complex_imags`

Returns the imaginary components of a complex array.

- C symbol: `terlan_ndarray_v1_complex_imags`
- NativeBoundary operation: `ndarray.array.complex_imags`
- Ownership: `value`

## `reshape`

Returns a shared reshape view when possible and otherwise a contiguous materialization.

- C symbol: `terlan_ndarray_v1_reshape`
- NativeBoundary operation: `ndarray.array.reshape`
- Ownership: `opaque_handle`

## `reshape`

Reshapes using explicit C, Fortran, or input-compatible traversal order.

- C symbol: `terlan_ndarray_v1_reshape_ordered`
- NativeBoundary operation: `ndarray.array.reshape_ordered`
- Ownership: `opaque_handle`

## `resize_in_place`

Resizes a uniquely owned contiguous array in place, preserving its flattened prefix and zero-initializing growth.

- C symbol: `terlan_ndarray_v1_resize_in_place`
- NativeBoundary operation: `ndarray.array.resize_in_place`
- Ownership: `borrowed_handle`

## `transpose`

Returns a shared strided view with two normalized axes exchanged.

- C symbol: `terlan_ndarray_v1_transpose`
- NativeBoundary operation: `ndarray.array.transpose`
- Ownership: `opaque_handle`

## `copy`

Returns an independently owned copy preserving source-compatible memory order.

- C symbol: `terlan_ndarray_v1_copy`
- NativeBoundary operation: `ndarray.array.copy`
- Ownership: `opaque_handle`

## `copy`

Returns an independently owned copy using C, Fortran, or source-compatible memory order.

- C symbol: `terlan_ndarray_v1_copy_ordered`
- NativeBoundary operation: `ndarray.array.copy_ordered`
- Ownership: `opaque_handle`

## `astype`

Returns a checked C-contiguous conversion to the requested dtype code.

- C symbol: `terlan_ndarray_v1_astype`
- NativeBoundary operation: `ndarray.array.astype`
- Ownership: `opaque_handle`

## `astype`

Converts dtype under explicit NumPy-compatible order, copy, and casting rules.

- C symbol: `terlan_ndarray_v1_astype_ordered`
- NativeBoundary operation: `ndarray.array.astype_ordered`
- Ownership: `opaque_handle`

## `contiguous`

Materializes the logical array into independently owned C-contiguous storage.

- C symbol: `terlan_ndarray_v1_contiguous`
- NativeBoundary operation: `ndarray.array.contiguous`
- Ownership: `opaque_handle`

## `index_axis`

Returns a shared view with one axis selected and removed; negative axes and indexes are normalized.

- C symbol: `terlan_ndarray_v1_index_axis`
- NativeBoundary operation: `ndarray.array.index_axis`
- Ownership: `opaque_handle`

## `slice_axis`

Returns a shared basic-slice view; the minimum Int value denotes an omitted start or stop.

- C symbol: `terlan_ndarray_v1_slice_axis`
- NativeBoundary operation: `ndarray.array.slice_axis`
- Ownership: `opaque_handle`

## `permute`

Returns a shared view using a complete normalized axis permutation.

- C symbol: `terlan_ndarray_v1_permute`
- NativeBoundary operation: `ndarray.array.permute`
- Ownership: `opaque_handle`

## `take`

Copies normalized integer selections along one axis, preserving duplicate indexes and order.

- C symbol: `terlan_ndarray_v1_take`
- NativeBoundary operation: `ndarray.array.take`
- Ownership: `opaque_handle`

## `take_mode`

Copies integer selections using raise, wrap, or clip index normalization.

- C symbol: `terlan_ndarray_v1_take_mode`
- NativeBoundary operation: `ndarray.array.take_mode`
- Ownership: `opaque_handle`

## `take_along_axis`

Takes one Int64 index at every output coordinate along a normalized axis.

- C symbol: `terlan_ndarray_v1_take_along_axis`
- NativeBoundary operation: `ndarray.array.take_along_axis`
- Ownership: `opaque_handle`

## `put`

Assigns scalar or matching values at normalized flattened indexes in order.

- C symbol: `terlan_ndarray_v1_put`
- NativeBoundary operation: `ndarray.array.put`
- Ownership: `opaque_handle`

## `put_along_axis`

Assigns scalar or shape-matched values using one Int64 index at each coordinate.

- C symbol: `terlan_ndarray_v1_put_along_axis`
- NativeBoundary operation: `ndarray.array.put_along_axis`
- Ownership: `opaque_handle`

## `compress`

Selects axis entries using a rank-one Bool mask, truncating the axis to the mask length.

- C symbol: `terlan_ndarray_v1_compress`
- NativeBoundary operation: `ndarray.array.compress`
- Ownership: `opaque_handle`

## `compress_flat`

Selects flattened logical values using a rank-one Bool mask, truncating to the mask length.

- C symbol: `terlan_ndarray_v1_compress_flat`
- NativeBoundary operation: `ndarray.array.compress_flat`
- Ownership: `opaque_handle`

## `mask_select`

Returns flattened values selected by an exact-shape Bool mask.

- C symbol: `terlan_ndarray_v1_mask_select`
- NativeBoundary operation: `ndarray.array.mask_select`
- Ownership: `opaque_handle`

## `mask_assign`

Assigns scalar or matching values where an exact-shape Bool mask is true.

- C symbol: `terlan_ndarray_v1_mask_assign`
- NativeBoundary operation: `ndarray.array.mask_assign`
- Ownership: `opaque_handle`

## `gather_coordinates`

Gathers flattened values from an Int64 coordinate matrix shaped [count, rank].

- C symbol: `terlan_ndarray_v1_gather_coordinates`
- NativeBoundary operation: `ndarray.array.gather_coordinates`
- Ownership: `opaque_handle`

## `put_coordinates`

Atomically assigns scalar or matching values through an Int64 coordinate matrix shaped [count, rank].

- C symbol: `terlan_ndarray_v1_put_coordinates`
- NativeBoundary operation: `ndarray.array.put_coordinates`
- Ownership: `opaque_handle`

## `item_bool`

Returns the truth value of an array containing exactly one element.

- C symbol: `terlan_ndarray_v1_item_bool`
- NativeBoundary operation: `ndarray.array.item_bool`
- Ownership: `borrowed_handle`

## `item_int`

Returns a truncated Int from an array containing exactly one real element.

- C symbol: `terlan_ndarray_v1_item_int`
- NativeBoundary operation: `ndarray.array.item_int`
- Ownership: `borrowed_handle`

## `get_bool`

Returns one Bool element after normalizing a complete index tuple.

- C symbol: `terlan_ndarray_v1_get_bool`
- NativeBoundary operation: `ndarray.array.get_bool`
- Ownership: `borrowed_handle`

## `get_int`

Returns one Int64 element after normalizing a complete index tuple.

- C symbol: `terlan_ndarray_v1_get_int`
- NativeBoundary operation: `ndarray.array.get_int`
- Ownership: `borrowed_handle`

## `get_float`

Returns one Float64 element after normalizing a complete index tuple.

- C symbol: `terlan_ndarray_v1_get_float`
- NativeBoundary operation: `ndarray.array.get_float`
- Ownership: `borrowed_handle`

## `set_bool`

Assigns one Bool element through a writable view and makes the mutation visible to its aliases.

- C symbol: `terlan_ndarray_v1_set_bool`
- NativeBoundary operation: `ndarray.array.set_bool`
- Ownership: `borrowed_handle`

## `set_int`

Assigns one Int64 element through a writable view and makes the mutation visible to its aliases.

- C symbol: `terlan_ndarray_v1_set_int`
- NativeBoundary operation: `ndarray.array.set_int`
- Ownership: `borrowed_handle`

## `set_float`

Assigns one Float64 element through a writable view and makes the mutation visible to its aliases.

- C symbol: `terlan_ndarray_v1_set_float`
- NativeBoundary operation: `ndarray.array.set_float`
- Ownership: `borrowed_handle`

## `add`

Adds broadcast-compatible numeric arrays with canonical dtype promotion.

- C symbol: `terlan_ndarray_v1_add`
- NativeBoundary operation: `ndarray.array.add`
- Ownership: `opaque_handle`

## `subtract`

Subtracts broadcast-compatible numeric arrays with canonical dtype promotion.

- C symbol: `terlan_ndarray_v1_subtract`
- NativeBoundary operation: `ndarray.array.subtract`
- Ownership: `opaque_handle`

## `multiply`

Multiplies broadcast-compatible numeric arrays with canonical dtype promotion.

- C symbol: `terlan_ndarray_v1_multiply`
- NativeBoundary operation: `ndarray.array.multiply`
- Ownership: `opaque_handle`

## `matmul`

Performs vector, matrix, and broadcast-batched matrix multiplication with numeric dtype promotion.

- C symbol: `terlan_ndarray_v1_matmul`
- NativeBoundary operation: `ndarray.array.matmul`
- Ownership: `opaque_handle`

## `dot`

Computes NumPy-compatible scalar multiplication, vector inner products, and generalized array dot products.

- C symbol: `terlan_ndarray_v1_dot`
- NativeBoundary operation: `ndarray.array.dot`
- Ownership: `opaque_handle`

## `inner`

Contracts the final dimensions of two arrays without conjugation.

- C symbol: `terlan_ndarray_v1_inner`
- NativeBoundary operation: `ndarray.array.inner`
- Ownership: `opaque_handle`

## `outer`

Computes the outer product of the flattened inputs.

- C symbol: `terlan_ndarray_v1_outer`
- NativeBoundary operation: `ndarray.array.outer`
- Ownership: `opaque_handle`

## `vdot`

Flattens both inputs and computes their conjugating vector dot product.

- C symbol: `terlan_ndarray_v1_vdot`
- NativeBoundary operation: `ndarray.array.vdot`
- Ownership: `opaque_handle`

## `trace`

Sums a selected diagonal over two distinct axes.

- C symbol: `terlan_ndarray_v1_trace`
- NativeBoundary operation: `ndarray.array.trace`
- Ownership: `opaque_handle`

## `determinant`

Computes a finite real square matrix determinant through maintained faer factorization.

- C symbol: `terlan_ndarray_v1_determinant`
- NativeBoundary operation: `ndarray.array.determinant`
- Ownership: `value`

## `solve`

Solves a finite real square system for a vector or matrix right-hand side through maintained faer factorization.

- C symbol: `terlan_ndarray_v1_solve`
- NativeBoundary operation: `ndarray.array.solve`
- Ownership: `opaque_handle`

## `inverse`

Computes a finite real square matrix inverse by solving against identity through maintained faer factorization.

- C symbol: `terlan_ndarray_v1_inverse`
- NativeBoundary operation: `ndarray.array.inverse`
- Ownership: `opaque_handle`

## `norm_l1`

Computes a vector one-norm or matrix induced one-norm.

- C symbol: `terlan_ndarray_v1_norm_l1`
- NativeBoundary operation: `ndarray.array.norm_l1`
- Ownership: `value`

## `norm_l2`

Computes a vector Euclidean norm or matrix spectral norm.

- C symbol: `terlan_ndarray_v1_norm_l2`
- NativeBoundary operation: `ndarray.array.norm_l2`
- Ownership: `value`

## `norm_max`

Computes a vector maximum norm or matrix induced infinity norm.

- C symbol: `terlan_ndarray_v1_norm_max`
- NativeBoundary operation: `ndarray.array.norm_max`
- Ownership: `value`

## `frobenius_norm`

Computes a vector Euclidean norm or matrix Frobenius norm.

- C symbol: `terlan_ndarray_v1_frobenius_norm`
- NativeBoundary operation: `ndarray.array.frobenius_norm`
- Ownership: `value`

## `nuclear_norm`

Computes the nuclear norm of a finite real matrix.

- C symbol: `terlan_ndarray_v1_nuclear_norm`
- NativeBoundary operation: `ndarray.array.nuclear_norm`
- Ownership: `value`

## `matrix_rank`

Computes numerical matrix rank; zero tolerance selects the deterministic default.

- C symbol: `terlan_ndarray_v1_matrix_rank`
- NativeBoundary operation: `ndarray.array.matrix_rank`
- Ownership: `value`

## `to_npy`

Encodes an array as deterministic NumPy .npy bytes.

- C symbol: `terlan_ndarray_v1_to_npy`
- NativeBoundary operation: `ndarray.array.to_npy`
- Ownership: `value`

## `from_npy`

Decodes bounded NumPy .npy bytes into an owned CPU array.

- C symbol: `terlan_ndarray_v1_from_npy`
- NativeBoundary operation: `ndarray.array.from_npy`
- Ownership: `opaque_handle`

## `save_npy`

Saves an array to an exact filesystem path in deterministic NumPy .npy format.

- C symbol: `terlan_ndarray_v1_save_npy`
- NativeBoundary operation: `ndarray.array.save_npy`
- Ownership: `value`

## `load_npy`

Loads a bounded NumPy .npy file from an exact filesystem path.

- C symbol: `terlan_ndarray_v1_load_npy`
- NativeBoundary operation: `ndarray.array.load_npy`
- Ownership: `opaque_handle`

## `serialize`

Encodes an array as a versioned checksummed package-native packet.

- C symbol: `terlan_ndarray_v1_serialize`
- NativeBoundary operation: `ndarray.array.serialize`
- Ownership: `value`

## `deserialize`

Decodes a bounded checksummed package-native packet into an owned CPU array.

- C symbol: `terlan_ndarray_v1_deserialize`
- NativeBoundary operation: `ndarray.array.deserialize`
- Ownership: `opaque_handle`

## `mmap_npy`

Opens a bounded NumPy file as a read-only or read-write memory-backed array.

- C symbol: `terlan_ndarray_v1_mmap_npy`
- NativeBoundary operation: `ndarray.array.mmap_npy`
- Ownership: `opaque_handle`

## `flush`

Flushes writable memory-backed storage and is a no-op for ordinary or read-only arrays.

- C symbol: `terlan_ndarray_v1_flush`
- NativeBoundary operation: `ndarray.array.flush`
- Ownership: `opaque_handle`

## `to_arrow`

Exports a compatible primitive array through official Arrow C Data ownership callbacks.

- C symbol: `terlan_ndarray_v1_to_arrow`
- NativeBoundary operation: `ndarray.array.to_arrow`
- Ownership: `opaque_handle`

## `from_arrow`

Imports one package-owned non-null Arrow C Data export as an owned array.

- C symbol: `terlan_ndarray_v1_from_arrow`
- NativeBoundary operation: `ndarray.array.from_arrow`
- Ownership: `opaque_handle`

## `split`

Splits an axis into equally sized shared views.

- C symbol: `terlan_ndarray_v1_split`
- NativeBoundary operation: `ndarray.array.split`
- Ownership: `opaque_handle`

## `array_split`

Splits an axis into uneven shared views with larger leading sections.

- C symbol: `terlan_ndarray_v1_array_split`
- NativeBoundary operation: `ndarray.array.array_split`
- Ownership: `opaque_handle`

## `split_at`

Splits an axis before each normalized index and preserves empty sections.

- C symbol: `terlan_ndarray_v1_split_at`
- NativeBoundary operation: `ndarray.array.split_at`
- Ownership: `opaque_handle`

## `count_nonzero`

Counts truthy values across the flattened array.

- C symbol: `terlan_ndarray_v1_count_nonzero`
- NativeBoundary operation: `ndarray.array.count_nonzero`
- Ownership: `opaque_handle`

## `count_nonzero_axes`

Counts nonzero elements over normalized axes with optional retained dimensions.

- C symbol: `terlan_ndarray_v1_count_nonzero_axes`
- NativeBoundary operation: `ndarray.array.count_nonzero_axes`
- Ownership: `opaque_handle`

## `flatnonzero`

Returns flattened indexes of truthy values as Int64.

- C symbol: `terlan_ndarray_v1_flatnonzero`
- NativeBoundary operation: `ndarray.array.flatnonzero`
- Ownership: `opaque_handle`

## `argwhere`

Returns one coordinate row for every truthy value.

- C symbol: `terlan_ndarray_v1_argwhere`
- NativeBoundary operation: `ndarray.array.argwhere`
- Ownership: `opaque_handle`

## `nonzero`

Returns one Int64 coordinate array per source axis.

- C symbol: `terlan_ndarray_v1_nonzero`
- NativeBoundary operation: `ndarray.array.nonzero`
- Ownership: `opaque_handle`

## `expand_dims`

Returns a shared view with one singleton axis inserted.

- C symbol: `terlan_ndarray_v1_expand_dims`
- NativeBoundary operation: `ndarray.array.expand_dims`
- Ownership: `opaque_handle`

## `expand_dims`

Returns a shared view with singleton axes inserted at final output positions.

- C symbol: `terlan_ndarray_v1_expand_dims_axes`
- NativeBoundary operation: `ndarray.array.expand_dims_axes`
- Ownership: `opaque_handle`

## `squeeze`

Returns a shared view with selected singleton axes removed; an empty list removes all singleton axes.

- C symbol: `terlan_ndarray_v1_squeeze`
- NativeBoundary operation: `ndarray.array.squeeze`
- Ownership: `opaque_handle`

## `flip`

Returns a shared negative-stride view with selected axes reversed.

- C symbol: `terlan_ndarray_v1_flip`
- NativeBoundary operation: `ndarray.array.flip`
- Ownership: `opaque_handle`

## `roll`

Returns an owned array cyclically shifted along one axis.

- C symbol: `terlan_ndarray_v1_roll`
- NativeBoundary operation: `ndarray.array.roll`
- Ownership: `opaque_handle`

## `roll`

Returns an owned array cyclically shifted after flattening in logical row-major order.

- C symbol: `terlan_ndarray_v1_roll_flat`
- NativeBoundary operation: `ndarray.array.roll_flat`
- Ownership: `opaque_handle`

## `roll`

Returns an owned array shifted by the same amount along every selected axis.

- C symbol: `terlan_ndarray_v1_roll_axes_by`
- NativeBoundary operation: `ndarray.array.roll_axes_by`
- Ownership: `opaque_handle`

## `roll`

Returns an owned array shifted along selected axes using one broadcast shift or one shift per axis.

- C symbol: `terlan_ndarray_v1_roll_axes`
- NativeBoundary operation: `ndarray.array.roll_axes`
- Ownership: `opaque_handle`

## `repeat`

Repeats each element a non-negative number of times along one axis.

- C symbol: `terlan_ndarray_v1_repeat`
- NativeBoundary operation: `ndarray.array.repeat`
- Ownership: `opaque_handle`

## `repeat_counts`

Repeats axis entries using one broadcast count or one count per source entry.

- C symbol: `terlan_ndarray_v1_repeat_counts`
- NativeBoundary operation: `ndarray.array.repeat_counts`
- Ownership: `opaque_handle`

## `repeat_flat`

Flattens in logical C order and repeats entries using broadcast or per-entry counts.

- C symbol: `terlan_ndarray_v1_repeat_flat`
- NativeBoundary operation: `ndarray.array.repeat_flat`
- Ownership: `opaque_handle`

## `tile`

Tiles an array according to trailing-axis repetition counts.

- C symbol: `terlan_ndarray_v1_tile`
- NativeBoundary operation: `ndarray.array.tile`
- Ownership: `opaque_handle`

## `concatenate`

Concatenates a nonempty list of shape-compatible arrays with canonical dtype promotion.

- C symbol: `terlan_ndarray_v1_concatenate`
- NativeBoundary operation: `ndarray.array.concatenate`
- Ownership: `opaque_handle`

## `stack`

Stacks a nonempty list of equal-shape arrays with canonical dtype promotion.

- C symbol: `terlan_ndarray_v1_stack`
- NativeBoundary operation: `ndarray.array.stack`
- Ownership: `opaque_handle`

## `sort`

Returns an axis-wise stable ascending copy using deterministic NaN and complex ordering.

- C symbol: `terlan_ndarray_v1_sort`
- NativeBoundary operation: `ndarray.array.sort`
- Ownership: `opaque_handle`

## `argsort`

Returns stable Int64 indices that order each axis slice ascending.

- C symbol: `terlan_ndarray_v1_argsort`
- NativeBoundary operation: `ndarray.array.argsort`
- Ownership: `opaque_handle`

## `sort_unstable`

Returns an ascending axis-wise copy without preserving equal-value order.

- C symbol: `terlan_ndarray_v1_sort_unstable`
- NativeBoundary operation: `ndarray.array.sort_unstable`
- Ownership: `opaque_handle`

## `argsort_unstable`

Returns ascending Int64 indexes without preserving equal-value order.

- C symbol: `terlan_ndarray_v1_argsort_unstable`
- NativeBoundary operation: `ndarray.array.argsort_unstable`
- Ownership: `opaque_handle`

## `sort_flat`

Returns a stable ascending one-dimensional copy.

- C symbol: `terlan_ndarray_v1_sort_flat`
- NativeBoundary operation: `ndarray.array.sort_flat`
- Ownership: `opaque_handle`

## `argsort_flat`

Returns stable Int64 indices ordering the flattened array.

- C symbol: `terlan_ndarray_v1_argsort_flat`
- NativeBoundary operation: `ndarray.array.argsort_flat`
- Ownership: `opaque_handle`

## `sort_inplace`

Sorts a writable array in place along one axis.

- C symbol: `terlan_ndarray_v1_sort_inplace`
- NativeBoundary operation: `ndarray.array.sort_inplace`
- Ownership: `value`

## `partition`

Returns an axis-wise copy satisfying the selected kth partition postcondition.

- C symbol: `terlan_ndarray_v1_partition`
- NativeBoundary operation: `ndarray.array.partition`
- Ownership: `opaque_handle`

## `argpartition`

Returns Int64 indexes satisfying the selected kth partition postcondition.

- C symbol: `terlan_ndarray_v1_argpartition`
- NativeBoundary operation: `ndarray.array.argpartition`
- Ownership: `opaque_handle`

## `partition`

Returns an axis-wise partition for every integer position in kths without mutating either input.

- C symbol: `terlan_ndarray_v1_partition_kths`
- NativeBoundary operation: `ndarray.array.partition_kths`
- Ownership: `opaque_handle`

## `argpartition`

Returns Int64 indexes satisfying every integer position in kths.

- C symbol: `terlan_ndarray_v1_argpartition_kths`
- NativeBoundary operation: `ndarray.array.argpartition_kths`
- Ownership: `opaque_handle`

## `partition_flat`

Returns a flattened partitioned copy for one selected position.

- C symbol: `terlan_ndarray_v1_partition_flat`
- NativeBoundary operation: `ndarray.array.partition_flat`
- Ownership: `opaque_handle`

## `argpartition_flat`

Returns flattened Int64 indexes satisfying one selected partition position.

- C symbol: `terlan_ndarray_v1_argpartition_flat`
- NativeBoundary operation: `ndarray.array.argpartition_flat`
- Ownership: `opaque_handle`

## `partition_flat`

Returns a flattened partitioned copy for every integer position in kths.

- C symbol: `terlan_ndarray_v1_partition_kths_flat`
- NativeBoundary operation: `ndarray.array.partition_kths_flat`
- Ownership: `opaque_handle`

## `argpartition_flat`

Returns flattened Int64 indexes satisfying every integer position in kths.

- C symbol: `terlan_ndarray_v1_argpartition_kths_flat`
- NativeBoundary operation: `ndarray.array.argpartition_kths_flat`
- Ownership: `opaque_handle`

## `partition_inplace`

Partitions a writable array in place along one axis for one selected position.

- C symbol: `terlan_ndarray_v1_partition_inplace`
- NativeBoundary operation: `ndarray.array.partition_inplace`
- Ownership: `value`

## `partition_inplace`

Partitions a writable array in place along one axis for every integer position in kths.

- C symbol: `terlan_ndarray_v1_partition_kths_inplace`
- NativeBoundary operation: `ndarray.array.partition_kths_inplace`
- Ownership: `value`

## `broadcast_to`

Returns a read-only zero-stride view broadcast to a compatible shape.

- C symbol: `terlan_ndarray_v1_broadcast_to`
- NativeBoundary operation: `ndarray.array.broadcast_to`
- Ownership: `opaque_handle`

## `true_divide`

Divides broadcast-compatible arrays using NumPy dtype promotion.

- C symbol: `terlan_ndarray_v1_true_divide`
- NativeBoundary operation: `ndarray.array.true_divide`
- Ownership: `opaque_handle`

## `floor_divide`

Floor-divides broadcast-compatible real arrays.

- C symbol: `terlan_ndarray_v1_floor_divide`
- NativeBoundary operation: `ndarray.array.floor_divide`
- Ownership: `opaque_handle`

## `remainder`

Returns the NumPy-sign remainder of broadcast-compatible real arrays.

- C symbol: `terlan_ndarray_v1_remainder`
- NativeBoundary operation: `ndarray.array.remainder`
- Ownership: `opaque_handle`

## `power`

Raises broadcast-compatible arrays elementwise.

- C symbol: `terlan_ndarray_v1_power`
- NativeBoundary operation: `ndarray.array.power`
- Ownership: `opaque_handle`

## `elementwise_minimum`

Selects broadcasted minima while propagating NaN values.

- C symbol: `terlan_ndarray_v1_elementwise_minimum`
- NativeBoundary operation: `ndarray.array.elementwise_minimum`
- Ownership: `opaque_handle`

## `elementwise_maximum`

Selects broadcasted maxima while propagating NaN values.

- C symbol: `terlan_ndarray_v1_elementwise_maximum`
- NativeBoundary operation: `ndarray.array.elementwise_maximum`
- Ownership: `opaque_handle`

## `fmin`

Selects broadcasted minima while preferring a non-NaN operand.

- C symbol: `terlan_ndarray_v1_fmin`
- NativeBoundary operation: `ndarray.array.fmin`
- Ownership: `opaque_handle`

## `fmax`

Selects broadcasted maxima while preferring a non-NaN operand.

- C symbol: `terlan_ndarray_v1_fmax`
- NativeBoundary operation: `ndarray.array.fmax`
- Ownership: `opaque_handle`

## `clip`

Clips values through broadcast-compatible lower and upper arrays.

- C symbol: `terlan_ndarray_v1_clip`
- NativeBoundary operation: `ndarray.array.clip`
- Ownership: `opaque_handle`

## `clip_lower`

Clips values through a broadcast-compatible lower bound.

- C symbol: `terlan_ndarray_v1_clip_lower`
- NativeBoundary operation: `ndarray.array.clip_lower`
- Ownership: `opaque_handle`

## `clip_upper`

Clips values through a broadcast-compatible upper bound.

- C symbol: `terlan_ndarray_v1_clip_upper`
- NativeBoundary operation: `ndarray.array.clip_upper`
- Ownership: `opaque_handle`

## `clip_into`

Clips into a writable destination after complete shape and cast validation.

- C symbol: `terlan_ndarray_v1_clip_into`
- NativeBoundary operation: `ndarray.array.clip_into`
- Ownership: `value`

## `clip_lower_into`

Clips a lower bound into a writable destination after complete validation.

- C symbol: `terlan_ndarray_v1_clip_lower_into`
- NativeBoundary operation: `ndarray.array.clip_lower_into`
- Ownership: `value`

## `clip_upper_into`

Clips an upper bound into a writable destination after complete validation.

- C symbol: `terlan_ndarray_v1_clip_upper_into`
- NativeBoundary operation: `ndarray.array.clip_upper_into`
- Ownership: `value`

## `clip_inplace`

Clips an array in place after complete validation.

- C symbol: `terlan_ndarray_v1_clip_inplace`
- NativeBoundary operation: `ndarray.array.clip_inplace`
- Ownership: `value`

## `clip_lower_inplace`

Applies a lower bound to an array in place after complete validation.

- C symbol: `terlan_ndarray_v1_clip_lower_inplace`
- NativeBoundary operation: `ndarray.array.clip_lower_inplace`
- Ownership: `value`

## `clip_upper_inplace`

Applies an upper bound to an array in place after complete validation.

- C symbol: `terlan_ndarray_v1_clip_upper_inplace`
- NativeBoundary operation: `ndarray.array.clip_upper_inplace`
- Ownership: `value`

## `equal`

Compares broadcast-compatible arrays for equality.

- C symbol: `terlan_ndarray_v1_equal`
- NativeBoundary operation: `ndarray.array.equal`
- Ownership: `opaque_handle`

## `not_equal`

Compares broadcast-compatible arrays for inequality.

- C symbol: `terlan_ndarray_v1_not_equal`
- NativeBoundary operation: `ndarray.array.not_equal`
- Ownership: `opaque_handle`

## `less`

Compares broadcast-compatible real arrays with less-than.

- C symbol: `terlan_ndarray_v1_less`
- NativeBoundary operation: `ndarray.array.less`
- Ownership: `opaque_handle`

## `less_equal`

Compares broadcast-compatible real arrays with less-than-or-equal.

- C symbol: `terlan_ndarray_v1_less_equal`
- NativeBoundary operation: `ndarray.array.less_equal`
- Ownership: `opaque_handle`

## `greater`

Compares broadcast-compatible real arrays with greater-than.

- C symbol: `terlan_ndarray_v1_greater`
- NativeBoundary operation: `ndarray.array.greater`
- Ownership: `opaque_handle`

## `greater_equal`

Compares broadcast-compatible real arrays with greater-than-or-equal.

- C symbol: `terlan_ndarray_v1_greater_equal`
- NativeBoundary operation: `ndarray.array.greater_equal`
- Ownership: `opaque_handle`

## `logical_and`

Computes elementwise truth-value conjunction with broadcasting.

- C symbol: `terlan_ndarray_v1_logical_and`
- NativeBoundary operation: `ndarray.array.logical_and`
- Ownership: `opaque_handle`

## `logical_or`

Computes elementwise truth-value disjunction with broadcasting.

- C symbol: `terlan_ndarray_v1_logical_or`
- NativeBoundary operation: `ndarray.array.logical_or`
- Ownership: `opaque_handle`

## `logical_xor`

Computes elementwise truth-value exclusive disjunction with broadcasting.

- C symbol: `terlan_ndarray_v1_logical_xor`
- NativeBoundary operation: `ndarray.array.logical_xor`
- Ownership: `opaque_handle`

## `bitwise_and`

Computes broadcast-compatible integer bitwise conjunction.

- C symbol: `terlan_ndarray_v1_bitwise_and`
- NativeBoundary operation: `ndarray.array.bitwise_and`
- Ownership: `opaque_handle`

## `bitwise_or`

Computes broadcast-compatible integer bitwise disjunction.

- C symbol: `terlan_ndarray_v1_bitwise_or`
- NativeBoundary operation: `ndarray.array.bitwise_or`
- Ownership: `opaque_handle`

## `bitwise_xor`

Computes broadcast-compatible integer bitwise exclusive disjunction.

- C symbol: `terlan_ndarray_v1_bitwise_xor`
- NativeBoundary operation: `ndarray.array.bitwise_xor`
- Ownership: `opaque_handle`

## `left_shift`

Shifts integer values left by validated elementwise counts.

- C symbol: `terlan_ndarray_v1_left_shift`
- NativeBoundary operation: `ndarray.array.left_shift`
- Ownership: `opaque_handle`

## `right_shift`

Shifts integer values right by validated elementwise counts.

- C symbol: `terlan_ndarray_v1_right_shift`
- NativeBoundary operation: `ndarray.array.right_shift`
- Ownership: `opaque_handle`

## `gcd`

Returns elementwise non-negative greatest common divisors.

- C symbol: `terlan_ndarray_v1_gcd`
- NativeBoundary operation: `ndarray.array.gcd`
- Ownership: `opaque_handle`

## `lcm`

Returns checked elementwise least common multiples.

- C symbol: `terlan_ndarray_v1_lcm`
- NativeBoundary operation: `ndarray.array.lcm`
- Ownership: `opaque_handle`

## `hypot`

Returns stable elementwise Euclidean magnitudes.

- C symbol: `terlan_ndarray_v1_hypot`
- NativeBoundary operation: `ndarray.array.hypot`
- Ownership: `opaque_handle`

## `arctan2`

Returns signed elementwise angles from y and x components.

- C symbol: `terlan_ndarray_v1_arctan2`
- NativeBoundary operation: `ndarray.array.arctan2`
- Ownership: `opaque_handle`

## `logaddexp`

Computes stable elementwise logarithms of summed exponentials.

- C symbol: `terlan_ndarray_v1_logaddexp`
- NativeBoundary operation: `ndarray.array.logaddexp`
- Ownership: `opaque_handle`

## `logaddexp2`

Computes stable base-two logarithms of summed powers of two.

- C symbol: `terlan_ndarray_v1_logaddexp2`
- NativeBoundary operation: `ndarray.array.logaddexp2`
- Ownership: `opaque_handle`

## `copysign`

Copies elementwise floating signs from the second operand.

- C symbol: `terlan_ndarray_v1_copysign`
- NativeBoundary operation: `ndarray.array.copysign`
- Ownership: `opaque_handle`

## `nextafter`

Returns the next representable values toward the second operand.

- C symbol: `terlan_ndarray_v1_nextafter`
- NativeBoundary operation: `ndarray.array.nextafter`
- Ownership: `opaque_handle`

## `is_close`

Returns a broadcast-compatible tolerance-comparison Bool mask.

- C symbol: `terlan_ndarray_v1_is_close`
- NativeBoundary operation: `ndarray.array.is_close`
- Ownership: `opaque_handle`

## `select_where`

Selects promoted branch values through a broadcast-compatible truth-value condition.

- C symbol: `terlan_ndarray_v1_where`
- NativeBoundary operation: `ndarray.array.select_where`
- Ownership: `opaque_handle`

## `negative`

Negates every numerical element.

- C symbol: `terlan_ndarray_v1_negative`
- NativeBoundary operation: `ndarray.array.negative`
- Ownership: `opaque_handle`

## `positive`

Returns an owned numerical copy while rejecting Bool input.

- C symbol: `terlan_ndarray_v1_positive`
- NativeBoundary operation: `ndarray.array.positive`
- Ownership: `opaque_handle`

## `absolute`

Returns elementwise magnitude, using real output for complex arrays.

- C symbol: `terlan_ndarray_v1_absolute`
- NativeBoundary operation: `ndarray.array.absolute`
- Ownership: `opaque_handle`

## `sign`

Returns the elementwise numerical sign.

- C symbol: `terlan_ndarray_v1_sign`
- NativeBoundary operation: `ndarray.array.sign`
- Ownership: `opaque_handle`

## `square`

Squares every numerical element.

- C symbol: `terlan_ndarray_v1_square`
- NativeBoundary operation: `ndarray.array.square`
- Ownership: `opaque_handle`

## `sqrt`

Returns the elementwise principal square root.

- C symbol: `terlan_ndarray_v1_sqrt`
- NativeBoundary operation: `ndarray.array.sqrt`
- Ownership: `opaque_handle`

## `reciprocal`

Returns one divided by each nonzero element.

- C symbol: `terlan_ndarray_v1_reciprocal`
- NativeBoundary operation: `ndarray.array.reciprocal`
- Ownership: `opaque_handle`

## `exp`

Returns the elementwise exponential.

- C symbol: `terlan_ndarray_v1_exp`
- NativeBoundary operation: `ndarray.array.exp`
- Ownership: `opaque_handle`

## `expm1`

Returns the elementwise exponential minus one.

- C symbol: `terlan_ndarray_v1_expm1`
- NativeBoundary operation: `ndarray.array.expm1`
- Ownership: `opaque_handle`

## `log`

Returns the elementwise natural logarithm.

- C symbol: `terlan_ndarray_v1_log`
- NativeBoundary operation: `ndarray.array.log`
- Ownership: `opaque_handle`

## `log1p`

Returns the elementwise natural logarithm of one plus each value.

- C symbol: `terlan_ndarray_v1_log1p`
- NativeBoundary operation: `ndarray.array.log1p`
- Ownership: `opaque_handle`

## `log2`

Returns the elementwise base-two logarithm.

- C symbol: `terlan_ndarray_v1_log2`
- NativeBoundary operation: `ndarray.array.log2`
- Ownership: `opaque_handle`

## `log10`

Returns the elementwise base-ten logarithm.

- C symbol: `terlan_ndarray_v1_log10`
- NativeBoundary operation: `ndarray.array.log10`
- Ownership: `opaque_handle`

## `sinh`

Returns the elementwise hyperbolic sine.

- C symbol: `terlan_ndarray_v1_sinh`
- NativeBoundary operation: `ndarray.array.sinh`
- Ownership: `opaque_handle`

## `cosh`

Returns the elementwise hyperbolic cosine.

- C symbol: `terlan_ndarray_v1_cosh`
- NativeBoundary operation: `ndarray.array.cosh`
- Ownership: `opaque_handle`

## `tanh`

Returns the elementwise hyperbolic tangent.

- C symbol: `terlan_ndarray_v1_tanh`
- NativeBoundary operation: `ndarray.array.tanh`
- Ownership: `opaque_handle`

## `arcsin`

Returns the elementwise inverse sine.

- C symbol: `terlan_ndarray_v1_arcsin`
- NativeBoundary operation: `ndarray.array.arcsin`
- Ownership: `opaque_handle`

## `arccos`

Returns the elementwise inverse cosine.

- C symbol: `terlan_ndarray_v1_arccos`
- NativeBoundary operation: `ndarray.array.arccos`
- Ownership: `opaque_handle`

## `arctan`

Returns the elementwise inverse tangent.

- C symbol: `terlan_ndarray_v1_arctan`
- NativeBoundary operation: `ndarray.array.arctan`
- Ownership: `opaque_handle`

## `arcsinh`

Returns the elementwise inverse hyperbolic sine.

- C symbol: `terlan_ndarray_v1_arcsinh`
- NativeBoundary operation: `ndarray.array.arcsinh`
- Ownership: `opaque_handle`

## `arccosh`

Returns the elementwise inverse hyperbolic cosine.

- C symbol: `terlan_ndarray_v1_arccosh`
- NativeBoundary operation: `ndarray.array.arccosh`
- Ownership: `opaque_handle`

## `arctanh`

Returns the elementwise inverse hyperbolic tangent.

- C symbol: `terlan_ndarray_v1_arctanh`
- NativeBoundary operation: `ndarray.array.arctanh`
- Ownership: `opaque_handle`

## `sin`

Returns the elementwise sine.

- C symbol: `terlan_ndarray_v1_sin`
- NativeBoundary operation: `ndarray.array.sin`
- Ownership: `opaque_handle`

## `cos`

Returns the elementwise cosine.

- C symbol: `terlan_ndarray_v1_cos`
- NativeBoundary operation: `ndarray.array.cos`
- Ownership: `opaque_handle`

## `tan`

Returns the elementwise tangent.

- C symbol: `terlan_ndarray_v1_tan`
- NativeBoundary operation: `ndarray.array.tan`
- Ownership: `opaque_handle`

## `floor`

Returns the elementwise floor for real arrays.

- C symbol: `terlan_ndarray_v1_floor`
- NativeBoundary operation: `ndarray.array.floor`
- Ownership: `opaque_handle`

## `ceil`

Returns the elementwise ceiling for real arrays.

- C symbol: `terlan_ndarray_v1_ceil`
- NativeBoundary operation: `ndarray.array.ceil`
- Ownership: `opaque_handle`

## `round`

Rounds every element to the nearest even value.

- C symbol: `terlan_ndarray_v1_round`
- NativeBoundary operation: `ndarray.array.round`
- Ownership: `opaque_handle`

## `logical_not`

Returns elementwise truth-value negation.

- C symbol: `terlan_ndarray_v1_logical_not`
- NativeBoundary operation: `ndarray.array.logical_not`
- Ownership: `opaque_handle`

## `is_nan`

Returns a Bool mask identifying NaN components.

- C symbol: `terlan_ndarray_v1_is_nan`
- NativeBoundary operation: `ndarray.array.is_nan`
- Ownership: `opaque_handle`

## `is_infinite`

Returns a Bool mask identifying infinite components.

- C symbol: `terlan_ndarray_v1_is_infinite`
- NativeBoundary operation: `ndarray.array.is_infinite`
- Ownership: `opaque_handle`

## `is_finite`

Returns a Bool mask identifying values with finite components.

- C symbol: `terlan_ndarray_v1_is_finite`
- NativeBoundary operation: `ndarray.array.is_finite`
- Ownership: `opaque_handle`

## `conjugate`

Returns the elementwise complex conjugate.

- C symbol: `terlan_ndarray_v1_conjugate`
- NativeBoundary operation: `ndarray.array.conjugate`
- Ownership: `opaque_handle`

## `exp2`

Returns two raised to each element.

- C symbol: `terlan_ndarray_v1_exp2`
- NativeBoundary operation: `ndarray.array.exp2`
- Ownership: `opaque_handle`

## `cbrt`

Returns real elementwise cube roots.

- C symbol: `terlan_ndarray_v1_cbrt`
- NativeBoundary operation: `ndarray.array.cbrt`
- Ownership: `opaque_handle`

## `degrees`

Converts real angles from radians to degrees.

- C symbol: `terlan_ndarray_v1_degrees`
- NativeBoundary operation: `ndarray.array.degrees`
- Ownership: `opaque_handle`

## `radians`

Converts real angles from degrees to radians.

- C symbol: `terlan_ndarray_v1_radians`
- NativeBoundary operation: `ndarray.array.radians`
- Ownership: `opaque_handle`

## `sign_bit`

Returns a Bool mask preserving negative-zero sign bits.

- C symbol: `terlan_ndarray_v1_sign_bit`
- NativeBoundary operation: `ndarray.array.sign_bit`
- Ownership: `opaque_handle`

## `frexp_all`

Returns mantissa and integer-exponent arrays.

- C symbol: `terlan_ndarray_v1_frexp_all`
- NativeBoundary operation: `ndarray.array.frexp_all`
- Ownership: `opaque_handle`

## `modf_all`

Returns fractional and integral-part arrays.

- C symbol: `terlan_ndarray_v1_modf_all`
- NativeBoundary operation: `ndarray.array.modf_all`
- Ownership: `opaque_handle`

## `sum`

Reduces normalized axes; an empty axis list reduces all axes.

- C symbol: `terlan_ndarray_v1_sum`
- NativeBoundary operation: `ndarray.array.sum`
- Ownership: `opaque_handle`

## `product`

Reduces normalized axes using NumPy-compatible product semantics.

- C symbol: `terlan_ndarray_v1_product`
- NativeBoundary operation: `ndarray.array.product`
- Ownership: `opaque_handle`

## `mean`

Reduces normalized axes using NumPy-compatible mean semantics.

- C symbol: `terlan_ndarray_v1_mean`
- NativeBoundary operation: `ndarray.array.mean`
- Ownership: `opaque_handle`

## `mean_where`

Returns means over elements selected by a broadcast-compatible truth mask.

- C symbol: `terlan_ndarray_v1_mean_where`
- NativeBoundary operation: `ndarray.array.mean_where`
- Ownership: `opaque_handle`

## `mean_as`

Returns means using an explicit result dtype.

- C symbol: `terlan_ndarray_v1_mean_as`
- NativeBoundary operation: `ndarray.array.mean_as`
- Ownership: `opaque_handle`

## `mean_into`

Writes means into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_mean_into`
- NativeBoundary operation: `ndarray.array.mean_into`
- Ownership: `value`

## `minimum`

Reduces normalized axes using NumPy-compatible minimum semantics.

- C symbol: `terlan_ndarray_v1_minimum`
- NativeBoundary operation: `ndarray.array.minimum`
- Ownership: `opaque_handle`

## `maximum`

Reduces normalized axes using NumPy-compatible maximum semantics.

- C symbol: `terlan_ndarray_v1_maximum`
- NativeBoundary operation: `ndarray.array.maximum`
- Ownership: `opaque_handle`

## `any`

Reduces normalized axes using NumPy-compatible any semantics.

- C symbol: `terlan_ndarray_v1_any`
- NativeBoundary operation: `ndarray.array.any`
- Ownership: `opaque_handle`

## `all`

Reduces normalized axes using NumPy-compatible all semantics.

- C symbol: `terlan_ndarray_v1_all`
- NativeBoundary operation: `ndarray.array.all`
- Ownership: `opaque_handle`

## `argmin`

Reduces normalized axes using NumPy-compatible argmin semantics.

- C symbol: `terlan_ndarray_v1_argmin`
- NativeBoundary operation: `ndarray.array.argmin`
- Ownership: `opaque_handle`

## `argmax`

Reduces normalized axes using NumPy-compatible argmax semantics.

- C symbol: `terlan_ndarray_v1_argmax`
- NativeBoundary operation: `ndarray.array.argmax`
- Ownership: `opaque_handle`

## `argmin_into`

Writes argmin indexes into an exactly shaped writable integer or Bool destination.

- C symbol: `terlan_ndarray_v1_argmin_into`
- NativeBoundary operation: `ndarray.array.argmin_into`
- Ownership: `value`

## `argmax_into`

Writes argmax indexes into an exactly shaped writable integer or Bool destination.

- C symbol: `terlan_ndarray_v1_argmax_into`
- NativeBoundary operation: `ndarray.array.argmax_into`
- Ownership: `value`

## `variance`

Reduces normalized axes using variance with an explicit non-negative correction.

- C symbol: `terlan_ndarray_v1_variance`
- NativeBoundary operation: `ndarray.array.variance`
- Ownership: `opaque_handle`

## `standard_deviation`

Reduces normalized axes using standard deviation with an explicit non-negative correction.

- C symbol: `terlan_ndarray_v1_standard_deviation`
- NativeBoundary operation: `ndarray.array.standard_deviation`
- Ownership: `opaque_handle`

## `variance_where`

Returns variances over elements selected by a broadcast-compatible truth mask.

- C symbol: `terlan_ndarray_v1_variance_where`
- NativeBoundary operation: `ndarray.array.variance_where`
- Ownership: `opaque_handle`

## `variance_as`

Returns variances using an explicit result dtype.

- C symbol: `terlan_ndarray_v1_variance_as`
- NativeBoundary operation: `ndarray.array.variance_as`
- Ownership: `opaque_handle`

## `variance_into`

Writes variances into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_variance_into`
- NativeBoundary operation: `ndarray.array.variance_into`
- Ownership: `value`

## `standard_deviation_where`

Returns standard deviations over elements selected by a broadcast-compatible truth mask.

- C symbol: `terlan_ndarray_v1_standard_deviation_where`
- NativeBoundary operation: `ndarray.array.standard_deviation_where`
- Ownership: `opaque_handle`

## `standard_deviation_as`

Returns standard deviations using an explicit result dtype.

- C symbol: `terlan_ndarray_v1_standard_deviation_as`
- NativeBoundary operation: `ndarray.array.standard_deviation_as`
- Ownership: `opaque_handle`

## `standard_deviation_into`

Writes standard deviations into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_standard_deviation_into`
- NativeBoundary operation: `ndarray.array.standard_deviation_into`
- Ownership: `value`

## `median`

Returns the linear-interpolated median across normalized axes.

- C symbol: `terlan_ndarray_v1_median`
- NativeBoundary operation: `ndarray.array.median`
- Ownership: `opaque_handle`

## `quantile`

Returns an axis reduction at a unit-interval quantile using an explicit interpolation policy.

- C symbol: `terlan_ndarray_v1_quantile`
- NativeBoundary operation: `ndarray.array.quantile`
- Ownership: `opaque_handle`

## `percentile`

Returns an axis reduction at a percentage quantile using an explicit interpolation policy.

- C symbol: `terlan_ndarray_v1_percentile`
- NativeBoundary operation: `ndarray.array.percentile`
- Ownership: `opaque_handle`

## `quantiles`

Returns multiple unit-interval quantiles with the quantile dimension first.

- C symbol: `terlan_ndarray_v1_quantiles`
- NativeBoundary operation: `ndarray.array.quantiles`
- Ownership: `opaque_handle`

## `percentiles`

Returns multiple percentage quantiles with the percentile dimension first.

- C symbol: `terlan_ndarray_v1_percentiles`
- NativeBoundary operation: `ndarray.array.percentiles`
- Ownership: `opaque_handle`

## `peak_to_peak`

Returns the maximum minus minimum across normalized axes.

- C symbol: `terlan_ndarray_v1_peak_to_peak`
- NativeBoundary operation: `ndarray.array.peak_to_peak`
- Ownership: `opaque_handle`

## `weighted_average`

Returns an axis average weighted by a same-shape array or matching rank-one vector.

- C symbol: `terlan_ndarray_v1_weighted_average`
- NativeBoundary operation: `ndarray.array.weighted_average`
- Ownership: `opaque_handle`

## `covariance`

Returns a covariance matrix with explicit variable orientation and degrees-of-freedom correction.

- C symbol: `terlan_ndarray_v1_covariance`
- NativeBoundary operation: `ndarray.array.covariance`
- Ownership: `opaque_handle`

## `correlation`

Returns the normalized covariance matrix for non-constant variables.

- C symbol: `terlan_ndarray_v1_correlation`
- NativeBoundary operation: `ndarray.array.correlation`
- Ownership: `opaque_handle`

## `nan_sum`

Returns sums while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_sum`
- NativeBoundary operation: `ndarray.array.nan_sum`
- Ownership: `opaque_handle`

## `nan_product`

Returns products while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_product`
- NativeBoundary operation: `ndarray.array.nan_product`
- Ownership: `opaque_handle`

## `nan_mean`

Returns means while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_mean`
- NativeBoundary operation: `ndarray.array.nan_mean`
- Ownership: `opaque_handle`

## `nan_minimum`

Returns minima while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_minimum`
- NativeBoundary operation: `ndarray.array.nan_minimum`
- Ownership: `opaque_handle`

## `nan_maximum`

Returns maxima while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_maximum`
- NativeBoundary operation: `ndarray.array.nan_maximum`
- Ownership: `opaque_handle`

## `nan_variance`

Returns variance while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_variance`
- NativeBoundary operation: `ndarray.array.nan_variance`
- Ownership: `opaque_handle`

## `nan_standard_deviation`

Returns standard deviation while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_standard_deviation`
- NativeBoundary operation: `ndarray.array.nan_standard_deviation`
- Ownership: `opaque_handle`

## `histogram`

Counts flattened values into equal-width bins over an explicit finite range.

- C symbol: `terlan_ndarray_v1_histogram`
- NativeBoundary operation: `ndarray.array.histogram`
- Ownership: `opaque_handle`

## `histogram_bin_edges`

Returns equal-width Float64 histogram edges for an explicit finite range.

- C symbol: `terlan_ndarray_v1_histogram_bin_edges`
- NativeBoundary operation: `ndarray.array.histogram_bin_edges`
- Ownership: `opaque_handle`

## `histogramdd`

Counts a [sample, dimension] matrix into an equal-width multidimensional histogram.

- C symbol: `terlan_ndarray_v1_histogramdd`
- NativeBoundary operation: `ndarray.array.histogramdd`
- Ownership: `opaque_handle`

## `bincount`

Accumulates finite weights by non-negative integer value.

- C symbol: `terlan_ndarray_v1_bincount`
- NativeBoundary operation: `ndarray.array.bincount`
- Ownership: `opaque_handle`

## `digitize`

Returns insertion-bin indexes against strictly increasing rank-one edges.

- C symbol: `terlan_ndarray_v1_digitize`
- NativeBoundary operation: `ndarray.array.digitize`
- Ownership: `opaque_handle`

## `difference`

Returns repeated forward differences along one normalized axis.

- C symbol: `terlan_ndarray_v1_difference`
- NativeBoundary operation: `ndarray.array.difference`
- Ownership: `opaque_handle`

## `gradient`

Returns central interior and first-order edge gradients for uniform spacing.

- C symbol: `terlan_ndarray_v1_gradient`
- NativeBoundary operation: `ndarray.array.gradient`
- Ownership: `opaque_handle`

## `unwrap`

Unwraps real periodic samples along one normalized axis.

- C symbol: `terlan_ndarray_v1_unwrap`
- NativeBoundary operation: `ndarray.array.unwrap`
- Ownership: `opaque_handle`

## `interpolate`

Linearly interpolates real values over strictly increasing rank-one coordinates.

- C symbol: `terlan_ndarray_v1_interpolate`
- NativeBoundary operation: `ndarray.array.interpolate`
- Ownership: `opaque_handle`

## `trapezoidal_integral`

Integrates along one axis using explicit rank-one coordinates and the trapezoidal rule.

- C symbol: `terlan_ndarray_v1_trapezoidal_integral`
- NativeBoundary operation: `ndarray.array.trapezoidal_integral`
- Ownership: `opaque_handle`

## `trapezoidal_integral_uniform`

Integrates along one axis using uniform finite spacing and the trapezoidal rule.

- C symbol: `terlan_ndarray_v1_trapezoidal_integral_uniform`
- NativeBoundary operation: `ndarray.array.trapezoidal_integral_uniform`
- Ownership: `opaque_handle`

## `fft`

Returns a complex FFT along one normalized axis.

- C symbol: `terlan_ndarray_v1_fft`
- NativeBoundary operation: `ndarray.array.fft`
- Ownership: `opaque_handle`

## `fft`

Returns a complex FFT after truncating or zero-padding one normalized axis to an explicit length.

- C symbol: `terlan_ndarray_v1_fft_sized`
- NativeBoundary operation: `ndarray.array.fft_sized`
- Ownership: `opaque_handle`

## `ifft`

Returns an inverse complex FFT along one normalized axis.

- C symbol: `terlan_ndarray_v1_ifft`
- NativeBoundary operation: `ndarray.array.ifft`
- Ownership: `opaque_handle`

## `ifft`

Returns an inverse complex FFT after truncating or zero-padding one normalized axis to an explicit length.

- C symbol: `terlan_ndarray_v1_ifft_sized`
- NativeBoundary operation: `ndarray.array.ifft_sized`
- Ownership: `opaque_handle`

## `fftn`

Returns complex FFTs across normalized distinct axes.

- C symbol: `terlan_ndarray_v1_fftn`
- NativeBoundary operation: `ndarray.array.fftn`
- Ownership: `opaque_handle`

## `ifftn`

Returns inverse complex FFTs across normalized distinct axes.

- C symbol: `terlan_ndarray_v1_ifftn`
- NativeBoundary operation: `ndarray.array.ifftn`
- Ownership: `opaque_handle`

## `rfft`

Returns non-negative bins from a real-input FFT.

- C symbol: `terlan_ndarray_v1_rfft`
- NativeBoundary operation: `ndarray.array.rfft`
- Ownership: `opaque_handle`

## `rfft`

Returns non-negative bins after truncating or zero-padding a real-input axis to an explicit length.

- C symbol: `terlan_ndarray_v1_rfft_sized`
- NativeBoundary operation: `ndarray.array.rfft_sized`
- Ownership: `opaque_handle`

## `irfft`

Reconstructs a real signal of explicit length from compact frequency bins.

- C symbol: `terlan_ndarray_v1_irfft`
- NativeBoundary operation: `ndarray.array.irfft`
- Ownership: `opaque_handle`

## `rfftn`

Returns compact real-input FFTs across normalized distinct axes.

- C symbol: `terlan_ndarray_v1_rfftn`
- NativeBoundary operation: `ndarray.array.rfftn`
- Ownership: `opaque_handle`

## `irfftn`

Reconstructs a real array from compact N-dimensional frequency bins.

- C symbol: `terlan_ndarray_v1_irfftn`
- NativeBoundary operation: `ndarray.array.irfftn`
- Ownership: `opaque_handle`

## `hfft`

Transforms a compact Hermitian signal to a real spectrum.

- C symbol: `terlan_ndarray_v1_hfft`
- NativeBoundary operation: `ndarray.array.hfft`
- Ownership: `opaque_handle`

## `ihfft`

Transforms a real spectrum to a compact Hermitian signal.

- C symbol: `terlan_ndarray_v1_ihfft`
- NativeBoundary operation: `ndarray.array.ihfft`
- Ownership: `opaque_handle`

## `ihfft`

Transforms an explicitly sized real spectrum to a compact Hermitian signal.

- C symbol: `terlan_ndarray_v1_ihfft_sized`
- NativeBoundary operation: `ndarray.array.ihfft_sized`
- Ownership: `opaque_handle`

## `fftfreq`

Returns signed discrete frequency bins.

- C symbol: `terlan_ndarray_v1_fftfreq`
- NativeBoundary operation: `ndarray.array.fftfreq`
- Ownership: `opaque_handle`

## `rfftfreq`

Returns non-negative real-transform frequency bins.

- C symbol: `terlan_ndarray_v1_rfftfreq`
- NativeBoundary operation: `ndarray.array.rfftfreq`
- Ownership: `opaque_handle`

## `fftshift`

Moves zero-frequency bins to centered positions on selected axes.

- C symbol: `terlan_ndarray_v1_fftshift`
- NativeBoundary operation: `ndarray.array.fftshift`
- Ownership: `opaque_handle`

## `ifftshift`

Reverses centered frequency-bin shifts on selected axes.

- C symbol: `terlan_ndarray_v1_ifftshift`
- NativeBoundary operation: `ndarray.array.ifftshift`
- Ownership: `opaque_handle`

## `searchsorted`

Returns stable left or right insertion indexes into one sorted rank-one array.

- C symbol: `terlan_ndarray_v1_searchsorted`
- NativeBoundary operation: `ndarray.array.searchsorted`
- Ownership: `opaque_handle`

## `searchsorted_with_sorter`

Returns insertion indexes using a validated Int64 permutation of an unsorted rank-one array.

- C symbol: `terlan_ndarray_v1_searchsorted_with_sorter`
- NativeBoundary operation: `ndarray.array.searchsorted_with_sorter`
- Ownership: `opaque_handle`

## `lexsort_keys`

Returns stable lexicographic indexes for a [key_count, item_count] matrix using the final key as primary.

- C symbol: `terlan_ndarray_v1_lexsort_keys`
- NativeBoundary operation: `ndarray.array.lexsort_keys`
- Ownership: `opaque_handle`

## `unique_all`

Returns sorted unique values, first indexes, inverse indexes, and occurrence counts.

- C symbol: `terlan_ndarray_v1_unique_all`
- NativeBoundary operation: `ndarray.array.unique_all`
- Ownership: `opaque_handle`

## `union`

Returns the sorted flattened union of two equal-dtype arrays.

- C symbol: `terlan_ndarray_v1_union`
- NativeBoundary operation: `ndarray.array.union`
- Ownership: `opaque_handle`

## `intersect`

Returns the sorted flattened intersection of two equal-dtype arrays.

- C symbol: `terlan_ndarray_v1_intersect`
- NativeBoundary operation: `ndarray.array.intersect`
- Ownership: `opaque_handle`

## `set_difference`

Returns sorted flattened values present only in the first array.

- C symbol: `terlan_ndarray_v1_set_difference`
- NativeBoundary operation: `ndarray.array.set_difference`
- Ownership: `opaque_handle`

## `set_xor`

Returns sorted flattened values present in exactly one input.

- C symbol: `terlan_ndarray_v1_set_xor`
- NativeBoundary operation: `ndarray.array.set_xor`
- Ownership: `opaque_handle`

## `is_in`

Returns a same-shape Bool mask indicating membership in the flattened test array.

- C symbol: `terlan_ndarray_v1_is_in`
- NativeBoundary operation: `ndarray.array.is_in`
- Ownership: `opaque_handle`

## `in1d`

Returns a flattened Bool membership mask with optional inversion.

- C symbol: `terlan_ndarray_v1_in1d`
- NativeBoundary operation: `ndarray.array.in1d`
- Ownership: `opaque_handle`

## `set_difference_unique`

Returns first-array values absent from the second while preserving order and duplicates.

- C symbol: `terlan_ndarray_v1_set_difference_unique`
- NativeBoundary operation: `ndarray.array.set_difference_unique`
- Ownership: `opaque_handle`

## `intersect_all`

Returns sorted intersections and their first flattened indexes in both inputs.

- C symbol: `terlan_ndarray_v1_intersect_all`
- NativeBoundary operation: `ndarray.array.intersect_all`
- Ownership: `opaque_handle`

## `unique_axis_all`

Returns lexicographically unique axis slices, first indexes, inverse indexes, and counts.

- C symbol: `terlan_ndarray_v1_unique_axis_all`
- NativeBoundary operation: `ndarray.array.unique_axis_all`
- Ownership: `opaque_handle`

## `ediff1d_edges`

Returns flattened consecutive differences with typed values prepended and appended.

- C symbol: `terlan_ndarray_v1_ediff1d_edges`
- NativeBoundary operation: `ndarray.array.ediff1d_edges`
- Ownership: `opaque_handle`

## `cumulative_sum`

Returns inclusive sums along one normalized axis using reduction accumulator promotion.

- C symbol: `terlan_ndarray_v1_cumulative_sum`
- NativeBoundary operation: `ndarray.array.cumulative_sum`
- Ownership: `opaque_handle`

## `cumulative_product`

Returns inclusive products along one normalized axis using reduction accumulator promotion.

- C symbol: `terlan_ndarray_v1_cumulative_product`
- NativeBoundary operation: `ndarray.array.cumulative_product`
- Ownership: `opaque_handle`

## `export_tensor_packet`

Exports a pointer-free tensor packet through a package-native DLPack lifecycle.

- C symbol: `terlan_ndarray_v1_export_tensor_packet`
- NativeBoundary operation: `ndarray.array.export_tensor_packet`
- Ownership: `value`

## `import_tensor_packet`

Imports a validated pointer-free tensor packet as an owned CPU array.

- C symbol: `terlan_ndarray_v1_import_tensor_packet`
- NativeBoundary operation: `ndarray.array.import_tensor_packet`
- Ownership: `opaque_handle`

## `choose`

Selects broadcast choice values with raise, wrap, or clip index handling.

- C symbol: `terlan_ndarray_v1_choose`
- NativeBoundary operation: `ndarray.array.choose`
- Ownership: `opaque_handle`

## `fill`

Fills the receiver in place from one scalar array using receiver dtype conversion.

- C symbol: `terlan_ndarray_v1_fill`
- NativeBoundary operation: `ndarray.array.fill`
- Ownership: `value`

## `resize`

Returns a resized logical-order array by repeating or truncating source values.

- C symbol: `terlan_ndarray_v1_resize`
- NativeBoundary operation: `ndarray.array.resize`
- Ownership: `opaque_handle`

## `any_where`

Reduces selected values using NumPy-compatible any semantics and a broadcast boolean mask.

- C symbol: `terlan_ndarray_v1_any_where`
- NativeBoundary operation: `ndarray.array.any_where`
- Ownership: `opaque_handle`

## `all_where`

Reduces selected values using NumPy-compatible all semantics and a broadcast boolean mask.

- C symbol: `terlan_ndarray_v1_all_where`
- NativeBoundary operation: `ndarray.array.all_where`
- Ownership: `opaque_handle`

## `choose_into`

Selects broadcast choice values and atomically writes them into an exactly shaped destination.

- C symbol: `terlan_ndarray_v1_choose_into`
- NativeBoundary operation: `ndarray.array.choose_into`
- Ownership: `value`

## `round_decimals`

Rounds every real or complex component to the requested decimal position using ties-to-even semantics.

- C symbol: `terlan_ndarray_v1_round_decimals`
- NativeBoundary operation: `ndarray.array.round_decimals`
- Ownership: `opaque_handle`

## `round_into`

Rounds values and atomically writes them into an exactly shaped same-kind destination.

- C symbol: `terlan_ndarray_v1_round_into`
- NativeBoundary operation: `ndarray.array.round_into`
- Ownership: `value`

## `conjugate_into`

Conjugates values and atomically writes them into an exactly shaped same-kind destination.

- C symbol: `terlan_ndarray_v1_conjugate_into`
- NativeBoundary operation: `ndarray.array.conjugate_into`
- Ownership: `value`

## `matmul_into`

Multiplies arrays and atomically writes the result into an exactly shaped, exact-dtype destination.

- C symbol: `terlan_ndarray_v1_matmul_into`
- NativeBoundary operation: `ndarray.array.matmul_into`
- Ownership: `value`

## `dot_into`

Computes a dot product into an exact-shape, exact-dtype, C-contiguous destination.

- C symbol: `terlan_ndarray_v1_dot_into`
- NativeBoundary operation: `ndarray.array.dot_into`
- Ownership: `value`

## `trace_into`

Computes a diagonal sum and atomically writes it into an exactly shaped destination.

- C symbol: `terlan_ndarray_v1_trace_into`
- NativeBoundary operation: `ndarray.array.trace_into`
- Ownership: `value`

## `dispose`

Disposes an owned array exactly once.

- C symbol: `terlan_ndarray_v1_delete`
- NativeBoundary operation: `ndarray.array.dispose`
- Ownership: `dispose_handle`

## `outer_into`

Writes the outer product into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_outer_into`
- NativeBoundary operation: `ndarray.array.outer_into`
- Ownership: `value`

## `sum_into`

Sums into an exactly shaped writable destination without broadcasting the reduction result.

- C symbol: `terlan_ndarray_v1_sum_into`
- NativeBoundary operation: `ndarray.array.sum_into`
- Ownership: `value`

## `product_into`

Multiplies into an exactly shaped writable destination without broadcasting the reduction result.

- C symbol: `terlan_ndarray_v1_product_into`
- NativeBoundary operation: `ndarray.array.product_into`
- Ownership: `value`

## `minimum_into`

Reduces minima into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_minimum_into`
- NativeBoundary operation: `ndarray.array.minimum_into`
- Ownership: `value`

## `maximum_into`

Reduces maxima into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_maximum_into`
- NativeBoundary operation: `ndarray.array.maximum_into`
- Ownership: `value`

## `sum_where`

Sums elements selected by a broadcast-compatible truth mask.

- C symbol: `terlan_ndarray_v1_sum_where`
- NativeBoundary operation: `ndarray.array.sum_where`
- Ownership: `opaque_handle`

## `sum_initial`

Sums elements using a one-element array as the initial accumulator.

- C symbol: `terlan_ndarray_v1_sum_initial`
- NativeBoundary operation: `ndarray.array.sum_initial`
- Ownership: `opaque_handle`

## `sum_where_initial`

Sums selected elements from an explicit one-element initial accumulator.

- C symbol: `terlan_ndarray_v1_sum_where_initial`
- NativeBoundary operation: `ndarray.array.sum_where_initial`
- Ownership: `opaque_handle`

## `minimum_initial`

Reduces minima using a one-element initial value, including empty lanes.

- C symbol: `terlan_ndarray_v1_minimum_initial`
- NativeBoundary operation: `ndarray.array.minimum_initial`
- Ownership: `opaque_handle`

## `maximum_initial`

Reduces maxima using a one-element initial value, including empty lanes.

- C symbol: `terlan_ndarray_v1_maximum_initial`
- NativeBoundary operation: `ndarray.array.maximum_initial`
- Ownership: `opaque_handle`

## `minimum_where`

Reduces minima selected by a mask and rejects selected empty lanes.

- C symbol: `terlan_ndarray_v1_minimum_where`
- NativeBoundary operation: `ndarray.array.minimum_where`
- Ownership: `opaque_handle`

## `maximum_where`

Reduces maxima selected by a mask and rejects selected empty lanes.

- C symbol: `terlan_ndarray_v1_maximum_where`
- NativeBoundary operation: `ndarray.array.maximum_where`
- Ownership: `opaque_handle`

## `minimum_where_initial`

Reduces selected minima from an explicit one-element initial value.

- C symbol: `terlan_ndarray_v1_minimum_where_initial`
- NativeBoundary operation: `ndarray.array.minimum_where_initial`
- Ownership: `opaque_handle`

## `maximum_where_initial`

Reduces selected maxima from an explicit one-element initial value.

- C symbol: `terlan_ndarray_v1_maximum_where_initial`
- NativeBoundary operation: `ndarray.array.maximum_where_initial`
- Ownership: `opaque_handle`

## `fmin_reduce`

Reduces minima while ignoring NaN operands unless a lane contains only NaNs.

- C symbol: `terlan_ndarray_v1_fmin_reduce`
- NativeBoundary operation: `ndarray.array.fmin_reduce`
- Ownership: `opaque_handle`

## `fmax_reduce`

Reduces maxima while ignoring NaN operands unless a lane contains only NaNs.

- C symbol: `terlan_ndarray_v1_fmax_reduce`
- NativeBoundary operation: `ndarray.array.fmax_reduce`
- Ownership: `opaque_handle`

## `bitwise_and_reduce`

Reduces integer or Bool elements with bitwise AND and a width-preserving all-bits identity.

- C symbol: `terlan_ndarray_v1_bitwise_and_reduce`
- NativeBoundary operation: `ndarray.array.bitwise_and_reduce`
- Ownership: `opaque_handle`

## `bitwise_or_reduce`

Reduces integer or Bool elements with bitwise OR and a zero identity.

- C symbol: `terlan_ndarray_v1_bitwise_or_reduce`
- NativeBoundary operation: `ndarray.array.bitwise_or_reduce`
- Ownership: `opaque_handle`

## `bitwise_xor_reduce`

Reduces integer or Bool elements with bitwise XOR and a zero identity.

- C symbol: `terlan_ndarray_v1_bitwise_xor_reduce`
- NativeBoundary operation: `ndarray.array.bitwise_xor_reduce`
- Ownership: `opaque_handle`

## `live_count`

Returns live native allocations for lifecycle auditing.

- C symbol: `terlan_ndarray_v1_live_count`
- NativeBoundary operation: `ndarray.array.live_count`
- Ownership: `value`
