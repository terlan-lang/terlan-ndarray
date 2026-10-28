# ndarray.Numpy

NumPy-compatible numerical completion operations.

## `lexsort`

Returns stable lexicographic indexes for stacked keys along one normalized key axis, using the final key as primary.

- C symbol: `terlan_ndarray_v1_lexsort_axis`
- NativeBoundary operation: `ndarray.numpy.lexsort`
- Ownership: `opaque_handle`

## `allclose`

Provides NumPy-compatible allclose behavior.

- C symbol: `terlan_ndarray_v1_allclose`
- NativeBoundary operation: `ndarray.numpy.allclose`
- Ownership: `value`

## `isclose`

Returns elementwise NumPy-compatible closeness.

- C symbol: `terlan_ndarray_v1_isclose`
- NativeBoundary operation: `ndarray.numpy.isclose`
- Ownership: `opaque_handle`

## `angle`

Provides NumPy-compatible angle behavior.

- C symbol: `terlan_ndarray_v1_angle`
- NativeBoundary operation: `ndarray.numpy.angle`
- Ownership: `opaque_handle`

## `array_equal`

Provides NumPy-compatible array equal behavior.

- C symbol: `terlan_ndarray_v1_array_equal`
- NativeBoundary operation: `ndarray.numpy.array_equal`
- Ownership: `value`

## `array_equal_nan`

Returns strict equality while treating paired NaNs as equal.

- C symbol: `terlan_ndarray_v1_array_equal_nan`
- NativeBoundary operation: `ndarray.numpy.array_equal_nan`
- Ownership: `value`

## `array_equiv`

Provides NumPy-compatible array equiv behavior.

- C symbol: `terlan_ndarray_v1_array_equiv`
- NativeBoundary operation: `ndarray.numpy.array_equiv`
- Ownership: `value`

## `asarray_chkfinite`

Provides NumPy-compatible asarray chkfinite behavior.

- C symbol: `terlan_ndarray_v1_asarray_chkfinite`
- NativeBoundary operation: `ndarray.numpy.asarray_chkfinite`
- Ownership: `opaque_handle`

## `asarray_chkfinite`

Returns a finite-only copy in an explicit dtype and memory order.

- C symbol: `terlan_ndarray_v1_asarray_chkfinite_as`
- NativeBoundary operation: `ndarray.numpy.asarray_chkfinite_as`
- Ownership: `opaque_handle`

## `asfarray`

Provides NumPy-compatible asfarray behavior.

- C symbol: `terlan_ndarray_v1_asfarray`
- NativeBoundary operation: `ndarray.numpy.asfarray`
- Ownership: `opaque_handle`

## `difference_prepend`

Returns repeated differences after prepending compatible values along one axis.

- C symbol: `terlan_ndarray_v1_difference_prepend`
- NativeBoundary operation: `ndarray.numpy.difference_prepend`
- Ownership: `opaque_handle`

## `difference_append`

Returns repeated differences after appending compatible values along one axis.

- C symbol: `terlan_ndarray_v1_difference_append`
- NativeBoundary operation: `ndarray.numpy.difference_append`
- Ownership: `opaque_handle`

## `difference_surround`

Returns repeated differences with compatible values on both ends of one axis.

- C symbol: `terlan_ndarray_v1_difference_surround`
- NativeBoundary operation: `ndarray.numpy.difference_surround`
- Ownership: `opaque_handle`

## `gradient_edge`

Returns a uniform-spacing gradient with first- or second-order boundaries.

- C symbol: `terlan_ndarray_v1_gradient_edge`
- NativeBoundary operation: `ndarray.numpy.gradient_edge`
- Ownership: `opaque_handle`

## `gradient_coordinates`

Returns a gradient along one axis using rank-one sample coordinates.

- C symbol: `terlan_ndarray_v1_gradient_coordinates`
- NativeBoundary operation: `ndarray.numpy.gradient_coordinates`
- Ownership: `opaque_handle`

## `gradient_all`

Returns one uniform-spacing gradient for every source axis.

- C symbol: `terlan_ndarray_v1_gradient_all`
- NativeBoundary operation: `ndarray.numpy.gradient_all`
- Ownership: `value`

## `gradient_uniform_axes`

Returns gradients for selected axes using one scalar spacing per axis.

- C symbol: `terlan_ndarray_v1_gradient_uniform_axes`
- NativeBoundary operation: `ndarray.numpy.gradient_uniform_axes`
- Ownership: `value`

## `gradient_axes`

Returns gradients for selected axes using scalar or coordinate spacing arrays.

- C symbol: `terlan_ndarray_v1_gradient_axes`
- NativeBoundary operation: `ndarray.numpy.gradient_axes`
- Ownership: `value`

## `unwrap_discontinuity`

Unwraps a real-valued signal using an explicit discontinuity threshold.

- C symbol: `terlan_ndarray_v1_unwrap_discontinuity`
- NativeBoundary operation: `ndarray.numpy.unwrap_discontinuity`
- Ownership: `opaque_handle`

## `unwrap_integer_period`

Unwraps using integral period arithmetic and preserves integral semantics.

- C symbol: `terlan_ndarray_v1_unwrap_integer_period`
- NativeBoundary operation: `ndarray.numpy.unwrap_integer_period`
- Ownership: `opaque_handle`

## `interpolate_default`

Interpolates with the first and last samples as the default boundaries.

- C symbol: `terlan_ndarray_v1_interpolate_default`
- NativeBoundary operation: `ndarray.numpy.interpolate_default`
- Ownership: `opaque_handle`

## `interpolate_boundaries`

Interpolates with explicit real or complex scalar boundary arrays.

- C symbol: `terlan_ndarray_v1_interpolate_boundaries`
- NativeBoundary operation: `ndarray.numpy.interpolate_boundaries`
- Ownership: `opaque_handle`

## `interpolate_periodic`

Interpolates over normalized and sorted periodic coordinates.

- C symbol: `terlan_ndarray_v1_interpolate_periodic`
- NativeBoundary operation: `ndarray.numpy.interpolate_periodic`
- Ownership: `opaque_handle`

## `bincount`

Counts occurrences of each nonnegative integer value.

- C symbol: `terlan_ndarray_v1_bincount_unweighted`
- NativeBoundary operation: `ndarray.numpy.bincount`
- Ownership: `opaque_handle`

## `average`

Provides NumPy-compatible average behavior.

- C symbol: `terlan_ndarray_v1_average`
- NativeBoundary operation: `ndarray.numpy.average`
- Ownership: `opaque_handle`

## `bartlett`

Provides NumPy-compatible bartlett behavior.

- C symbol: `terlan_ndarray_v1_bartlett`
- NativeBoundary operation: `ndarray.numpy.bartlett`
- Ownership: `opaque_handle`

## `blackman`

Provides NumPy-compatible blackman behavior.

- C symbol: `terlan_ndarray_v1_blackman`
- NativeBoundary operation: `ndarray.numpy.blackman`
- Ownership: `opaque_handle`

## `hamming`

Provides NumPy-compatible hamming behavior.

- C symbol: `terlan_ndarray_v1_hamming`
- NativeBoundary operation: `ndarray.numpy.hamming`
- Ownership: `opaque_handle`

## `hanning`

Provides NumPy-compatible hanning behavior.

- C symbol: `terlan_ndarray_v1_hanning`
- NativeBoundary operation: `ndarray.numpy.hanning`
- Ownership: `opaque_handle`

## `kaiser`

Provides NumPy-compatible kaiser behavior.

- C symbol: `terlan_ndarray_v1_kaiser`
- NativeBoundary operation: `ndarray.numpy.kaiser`
- Ownership: `opaque_handle`

## `bitwise_not`

Provides NumPy-compatible bitwise not behavior.

- C symbol: `terlan_ndarray_v1_bitwise_not`
- NativeBoundary operation: `ndarray.numpy.bitwise_not`
- Ownership: `opaque_handle`

## `convolve`

Provides NumPy-compatible convolve behavior.

- C symbol: `terlan_ndarray_v1_convolve`
- NativeBoundary operation: `ndarray.numpy.convolve`
- Ownership: `opaque_handle`

## `correlate`

Provides NumPy-compatible correlate behavior.

- C symbol: `terlan_ndarray_v1_correlate`
- NativeBoundary operation: `ndarray.numpy.correlate`
- Ownership: `opaque_handle`

## `divmod`

Provides NumPy-compatible divmod behavior.

- C symbol: `terlan_ndarray_v1_divmod`
- NativeBoundary operation: `ndarray.numpy.divmod`
- Ownership: `opaque_handle`

## `fix`

Provides NumPy-compatible fix behavior.

- C symbol: `terlan_ndarray_v1_fix`
- NativeBoundary operation: `ndarray.numpy.fix`
- Ownership: `opaque_handle`

## `float_power`

Provides NumPy-compatible float power behavior.

- C symbol: `terlan_ndarray_v1_float_power`
- NativeBoundary operation: `ndarray.numpy.float_power`
- Ownership: `opaque_handle`

## `fmod`

Provides NumPy-compatible fmod behavior.

- C symbol: `terlan_ndarray_v1_fmod`
- NativeBoundary operation: `ndarray.numpy.fmod`
- Ownership: `opaque_handle`

## `heaviside`

Provides NumPy-compatible heaviside behavior.

- C symbol: `terlan_ndarray_v1_heaviside`
- NativeBoundary operation: `ndarray.numpy.heaviside`
- Ownership: `opaque_handle`

## `i0`

Provides NumPy-compatible i0 behavior.

- C symbol: `terlan_ndarray_v1_i0`
- NativeBoundary operation: `ndarray.numpy.i0`
- Ownership: `opaque_handle`

## `real`

Provides NumPy-compatible real behavior.

- C symbol: `terlan_ndarray_v1_real`
- NativeBoundary operation: `ndarray.numpy.real`
- Ownership: `opaque_handle`

## `imag`

Provides NumPy-compatible imag behavior.

- C symbol: `terlan_ndarray_v1_imag`
- NativeBoundary operation: `ndarray.numpy.imag`
- Ownership: `opaque_handle`

## `iscomplex`

Provides NumPy-compatible iscomplex behavior.

- C symbol: `terlan_ndarray_v1_iscomplex`
- NativeBoundary operation: `ndarray.numpy.iscomplex`
- Ownership: `opaque_handle`

## `isreal`

Provides NumPy-compatible isreal behavior.

- C symbol: `terlan_ndarray_v1_isreal`
- NativeBoundary operation: `ndarray.numpy.isreal`
- Ownership: `opaque_handle`

## `iscomplexobj`

Provides NumPy-compatible iscomplexobj behavior.

- C symbol: `terlan_ndarray_v1_iscomplexobj`
- NativeBoundary operation: `ndarray.numpy.iscomplexobj`
- Ownership: `value`

## `isrealobj`

Provides NumPy-compatible isrealobj behavior.

- C symbol: `terlan_ndarray_v1_isrealobj`
- NativeBoundary operation: `ndarray.numpy.isrealobj`
- Ownership: `value`

## `isneginf`

Provides NumPy-compatible isneginf behavior.

- C symbol: `terlan_ndarray_v1_isneginf`
- NativeBoundary operation: `ndarray.numpy.isneginf`
- Ownership: `opaque_handle`

## `isposinf`

Provides NumPy-compatible isposinf behavior.

- C symbol: `terlan_ndarray_v1_isposinf`
- NativeBoundary operation: `ndarray.numpy.isposinf`
- Ownership: `opaque_handle`

## `ldexp`

Provides NumPy-compatible ldexp behavior.

- C symbol: `terlan_ndarray_v1_ldexp`
- NativeBoundary operation: `ndarray.numpy.ldexp`
- Ownership: `opaque_handle`

## `nan_to_num`

Provides NumPy-compatible nan to num behavior.

- C symbol: `terlan_ndarray_v1_nan_to_num`
- NativeBoundary operation: `ndarray.numpy.nan_to_num`
- Ownership: `opaque_handle`

## `real_if_close`

Provides NumPy-compatible real if close behavior.

- C symbol: `terlan_ndarray_v1_real_if_close`
- NativeBoundary operation: `ndarray.numpy.real_if_close`
- Ownership: `opaque_handle`

## `sinc`

Provides NumPy-compatible sinc behavior.

- C symbol: `terlan_ndarray_v1_sinc`
- NativeBoundary operation: `ndarray.numpy.sinc`
- Ownership: `opaque_handle`

## `spacing`

Provides NumPy-compatible spacing behavior.

- C symbol: `terlan_ndarray_v1_spacing`
- NativeBoundary operation: `ndarray.numpy.spacing`
- Ownership: `opaque_handle`

## `sort_complex`

Provides NumPy-compatible sort complex behavior.

- C symbol: `terlan_ndarray_v1_sort_complex`
- NativeBoundary operation: `ndarray.numpy.sort_complex`
- Ownership: `opaque_handle`

## `trim_zeros`

Provides NumPy-compatible trim zeros behavior.

- C symbol: `terlan_ndarray_v1_trim_zeros`
- NativeBoundary operation: `ndarray.numpy.trim_zeros`
- Ownership: `opaque_handle`

## `unique`

Provides NumPy-compatible unique behavior.

- C symbol: `terlan_ndarray_v1_unique`
- NativeBoundary operation: `ndarray.numpy.unique`
- Ownership: `opaque_handle`

## `invert`

Provides NumPy-compatible invert behavior.

- C symbol: `terlan_ndarray_v1_invert`
- NativeBoundary operation: `ndarray.numpy.invert`
- Ownership: `opaque_handle`

## `trunc`

Provides NumPy-compatible trunc behavior.

- C symbol: `terlan_ndarray_v1_trunc`
- NativeBoundary operation: `ndarray.numpy.trunc`
- Ownership: `opaque_handle`

## `frombuffer`

Provides NumPy-compatible frombuffer behavior.

- C symbol: `terlan_ndarray_v1_frombuffer`
- NativeBoundary operation: `ndarray.numpy.frombuffer`
- Ownership: `opaque_handle`

## `nanargmax`

Provides NumPy-compatible nanargmax behavior.

- C symbol: `terlan_ndarray_v1_nanargmax`
- NativeBoundary operation: `ndarray.numpy.nanargmax`
- Ownership: `value`

## `nanargmin`

Provides NumPy-compatible nanargmin behavior.

- C symbol: `terlan_ndarray_v1_nanargmin`
- NativeBoundary operation: `ndarray.numpy.nanargmin`
- Ownership: `value`

## `nancumprod`

Provides NumPy-compatible nancumprod behavior.

- C symbol: `terlan_ndarray_v1_nancumprod`
- NativeBoundary operation: `ndarray.numpy.nancumprod`
- Ownership: `opaque_handle`

## `nancumsum`

Provides NumPy-compatible nancumsum behavior.

- C symbol: `terlan_ndarray_v1_nancumsum`
- NativeBoundary operation: `ndarray.numpy.nancumsum`
- Ownership: `opaque_handle`

## `nanmedian`

Provides NumPy-compatible nanmedian behavior.

- C symbol: `terlan_ndarray_v1_nanmedian`
- NativeBoundary operation: `ndarray.numpy.nanmedian`
- Ownership: `opaque_handle`

## `nanpercentile`

Provides NumPy-compatible nanpercentile behavior.

- C symbol: `terlan_ndarray_v1_nanpercentile`
- NativeBoundary operation: `ndarray.numpy.nanpercentile`
- Ownership: `opaque_handle`

## `nanquantile`

Provides NumPy-compatible nanquantile behavior.

- C symbol: `terlan_ndarray_v1_nanquantile`
- NativeBoundary operation: `ndarray.numpy.nanquantile`
- Ownership: `opaque_handle`

## `nanquantile_axes`

Returns one NaN-skipping quantile across normalized axes.

- C symbol: `terlan_ndarray_v1_nanquantile_axes`
- NativeBoundary operation: `ndarray.numpy.nanquantile_axes`
- Ownership: `opaque_handle`

## `nanpercentile_axes`

Returns one NaN-skipping percentile across normalized axes.

- C symbol: `terlan_ndarray_v1_nanpercentile_axes`
- NativeBoundary operation: `ndarray.numpy.nanpercentile_axes`
- Ownership: `opaque_handle`

## `nanquantiles`

Returns multiple NaN-skipping quantiles with the quantile dimension first.

- C symbol: `terlan_ndarray_v1_nanquantiles`
- NativeBoundary operation: `ndarray.numpy.nanquantiles`
- Ownership: `opaque_handle`

## `nanpercentiles`

Returns multiple NaN-skipping percentiles with the percentile dimension first.

- C symbol: `terlan_ndarray_v1_nanpercentiles`
- NativeBoundary operation: `ndarray.numpy.nanpercentiles`
- Ownership: `opaque_handle`

## `nanmedian_axes`

Returns a NaN-skipping median across normalized axes.

- C symbol: `terlan_ndarray_v1_nanmedian_axes`
- NativeBoundary operation: `ndarray.numpy.nanmedian_axes`
- Ownership: `opaque_handle`

## `median_into`

Atomically writes a median into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_median_into`
- NativeBoundary operation: `ndarray.numpy.median_into`
- Ownership: `value`

## `nanmedian_into`

Atomically writes a NaN-skipping median into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_nanmedian_into`
- NativeBoundary operation: `ndarray.numpy.nanmedian_into`
- Ownership: `value`

## `quantile_into`

Atomically writes one quantile into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_quantile_into`
- NativeBoundary operation: `ndarray.numpy.quantile_into`
- Ownership: `value`

## `percentile_into`

Atomically writes one percentile into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_percentile_into`
- NativeBoundary operation: `ndarray.numpy.percentile_into`
- Ownership: `value`

## `quantiles_into`

Atomically writes multiple quantiles into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_quantiles_into`
- NativeBoundary operation: `ndarray.numpy.quantiles_into`
- Ownership: `value`

## `percentiles_into`

Atomically writes multiple percentiles into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_percentiles_into`
- NativeBoundary operation: `ndarray.numpy.percentiles_into`
- Ownership: `value`

## `nanquantile_into`

Atomically writes one NaN-skipping quantile into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_nanquantile_into`
- NativeBoundary operation: `ndarray.numpy.nanquantile_into`
- Ownership: `value`

## `nanpercentile_into`

Atomically writes one NaN-skipping percentile into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_nanpercentile_into`
- NativeBoundary operation: `ndarray.numpy.nanpercentile_into`
- Ownership: `value`

## `nanquantiles_into`

Atomically writes multiple NaN-skipping quantiles into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_nanquantiles_into`
- NativeBoundary operation: `ndarray.numpy.nanquantiles_into`
- Ownership: `value`

## `nanpercentiles_into`

Atomically writes multiple NaN-skipping percentiles into an exactly shaped writable destination.

- C symbol: `terlan_ndarray_v1_nanpercentiles_into`
- NativeBoundary operation: `ndarray.numpy.nanpercentiles_into`
- Ownership: `value`

## `packbits`

Provides NumPy-compatible packbits behavior.

- C symbol: `terlan_ndarray_v1_packbits`
- NativeBoundary operation: `ndarray.numpy.packbits`
- Ownership: `opaque_handle`

## `unpackbits`

Provides NumPy-compatible unpackbits behavior.

- C symbol: `terlan_ndarray_v1_unpackbits`
- NativeBoundary operation: `ndarray.numpy.unpackbits`
- Ownership: `opaque_handle`

## `unpackbits_all`

Unpacks every available flattened UInt8 bit.

- C symbol: `terlan_ndarray_v1_unpackbits_all`
- NativeBoundary operation: `ndarray.numpy.unpackbits_all`
- Ownership: `opaque_handle`

## `packbits_axis`

Packs truth values into UInt8 bytes along one normalized axis.

- C symbol: `terlan_ndarray_v1_packbits_axis`
- NativeBoundary operation: `ndarray.numpy.packbits_axis`
- Ownership: `opaque_handle`

## `unpackbits_axis`

Unpacks a counted number of UInt8 bits along one normalized axis.

- C symbol: `terlan_ndarray_v1_unpackbits_axis`
- NativeBoundary operation: `ndarray.numpy.unpackbits_axis`
- Ownership: `opaque_handle`

## `unpackbits_axis_all`

Unpacks every available UInt8 bit along one normalized axis.

- C symbol: `terlan_ndarray_v1_unpackbits_axis_all`
- NativeBoundary operation: `ndarray.numpy.unpackbits_axis_all`
- Ownership: `opaque_handle`

## `alen`

Provides NumPy-compatible alen behavior.

- C symbol: `terlan_ndarray_v1_alen`
- NativeBoundary operation: `ndarray.numpy.alen`
- Ownership: `value`

## `append`

Provides NumPy-compatible append behavior.

- C symbol: `terlan_ndarray_v1_append`
- NativeBoundary operation: `ndarray.numpy.append`
- Ownership: `opaque_handle`

## `broadcast_arrays`

Provides NumPy-compatible broadcast arrays behavior.

- C symbol: `terlan_ndarray_v1_broadcast_arrays`
- NativeBoundary operation: `ndarray.numpy.broadcast_arrays`
- Ownership: `opaque_handle`

## `broadcast_shapes`

Provides NumPy-compatible broadcast shapes behavior.

- C symbol: `terlan_ndarray_v1_broadcast_shapes`
- NativeBoundary operation: `ndarray.numpy.broadcast_shapes`
- Ownership: `opaque_handle`

## `as_strided`

Returns a read-only shared view with an explicit shape and element strides.

- C symbol: `terlan_ndarray_v1_as_strided`
- NativeBoundary operation: `ndarray.numpy.as_strided`
- Ownership: `opaque_handle`

## `sliding_window_view`

Returns a read-only shared sliding-window view, using every axis when axes is empty.

- C symbol: `terlan_ndarray_v1_sliding_window_view`
- NativeBoundary operation: `ndarray.numpy.sliding_window_view`
- Ownership: `opaque_handle`

## `column_stack`

Provides NumPy-compatible column stack behavior.

- C symbol: `terlan_ndarray_v1_column_stack`
- NativeBoundary operation: `ndarray.numpy.column_stack`
- Ownership: `opaque_handle`

## `copyto`

Provides NumPy-compatible copyto behavior.

- C symbol: `terlan_ndarray_v1_copyto`
- NativeBoundary operation: `ndarray.numpy.copyto`
- Ownership: `value`

## `copyto`

Broadcasts and copies values under an explicit NumPy-compatible casting rule.

- C symbol: `terlan_ndarray_v1_copyto_casting`
- NativeBoundary operation: `ndarray.numpy.copyto_casting`
- Ownership: `value`

## `copyto`

Broadcasts a source and truth mask before copying selected values under an explicit casting rule.

- C symbol: `terlan_ndarray_v1_copyto_where`
- NativeBoundary operation: `ndarray.numpy.copyto_where`
- Ownership: `value`

## `delete`

Provides NumPy-compatible delete behavior.

- C symbol: `terlan_ndarray_v1_delete_indexes`
- NativeBoundary operation: `ndarray.numpy.delete`
- Ownership: `opaque_handle`

## `delete`

Deletes normalized integer indexes along one axis.

- C symbol: `terlan_ndarray_v1_delete_indexes_axis`
- NativeBoundary operation: `ndarray.numpy.delete_axis`
- Ownership: `opaque_handle`

## `delete`

Deletes positions selected by an exact-length Bool mask along one axis.

- C symbol: `terlan_ndarray_v1_delete_mask_axis`
- NativeBoundary operation: `ndarray.numpy.delete_mask_axis`
- Ownership: `opaque_handle`

## `diag`

Extracts a matrix diagonal view or constructs a dtype-preserving diagonal matrix from a vector.

- C symbol: `terlan_ndarray_v1_diag`
- NativeBoundary operation: `ndarray.numpy.diag`
- Ownership: `opaque_handle`

## `diag_indices`

Provides NumPy-compatible diag indices behavior.

- C symbol: `terlan_ndarray_v1_diag_indices`
- NativeBoundary operation: `ndarray.numpy.diag_indices`
- Ownership: `opaque_handle`

## `diag_indices_from`

Provides NumPy-compatible diag indices from behavior.

- C symbol: `terlan_ndarray_v1_diag_indices_from`
- NativeBoundary operation: `ndarray.numpy.diag_indices_from`
- Ownership: `opaque_handle`

## `dsplit`

Provides NumPy-compatible dsplit behavior.

- C symbol: `terlan_ndarray_v1_dsplit`
- NativeBoundary operation: `ndarray.numpy.dsplit`
- Ownership: `opaque_handle`

## `dstack`

Provides NumPy-compatible dstack behavior.

- C symbol: `terlan_ndarray_v1_dstack`
- NativeBoundary operation: `ndarray.numpy.dstack`
- Ownership: `opaque_handle`

## `ediff1d`

Provides NumPy-compatible ediff1d behavior.

- C symbol: `terlan_ndarray_v1_ediff1d`
- NativeBoundary operation: `ndarray.numpy.ediff1d`
- Ownership: `opaque_handle`

## `full`

Provides NumPy-compatible full behavior.

- C symbol: `terlan_ndarray_v1_full`
- NativeBoundary operation: `ndarray.numpy.full`
- Ownership: `opaque_handle`

## `full_like`

Provides NumPy-compatible full like behavior.

- C symbol: `terlan_ndarray_v1_full_like`
- NativeBoundary operation: `ndarray.numpy.full_like`
- Ownership: `opaque_handle`

## `histogram`

Builds a one-dimensional histogram with an inferred finite range.

- C symbol: `terlan_ndarray_v1_numpy_histogram`
- NativeBoundary operation: `ndarray.numpy.histogram`
- Ownership: `opaque_handle`

## `histogram_range`

Builds a one-dimensional histogram over an explicit finite range.

- C symbol: `terlan_ndarray_v1_numpy_histogram_range`
- NativeBoundary operation: `ndarray.numpy.histogram_range`
- Ownership: `opaque_handle`

## `histogram_edges`

Builds a one-dimensional histogram from explicit increasing edges.

- C symbol: `terlan_ndarray_v1_numpy_histogram_edges`
- NativeBoundary operation: `ndarray.numpy.histogram_edges`
- Ownership: `opaque_handle`

## `histogram_weighted`

Builds a weighted histogram with an inferred finite range.

- C symbol: `terlan_ndarray_v1_numpy_histogram_weighted`
- NativeBoundary operation: `ndarray.numpy.histogram_weighted`
- Ownership: `opaque_handle`

## `histogram_weighted_range`

Builds a weighted histogram over an explicit finite range.

- C symbol: `terlan_ndarray_v1_numpy_histogram_weighted_range`
- NativeBoundary operation: `ndarray.numpy.histogram_weighted_range`
- Ownership: `opaque_handle`

## `histogram_weighted_edges`

Builds a weighted histogram from explicit increasing edges.

- C symbol: `terlan_ndarray_v1_numpy_histogram_weighted_edges`
- NativeBoundary operation: `ndarray.numpy.histogram_weighted_edges`
- Ownership: `opaque_handle`

## `histogram_estimator`

Builds a histogram using a typed automatic bin estimator.

- C symbol: `terlan_ndarray_v1_numpy_histogram_estimator`
- NativeBoundary operation: `ndarray.numpy.histogram_estimator`
- Ownership: `opaque_handle`

## `histogram_estimator_range`

Builds an estimator-selected histogram over an explicit finite range.

- C symbol: `terlan_ndarray_v1_numpy_histogram_estimator_range`
- NativeBoundary operation: `ndarray.numpy.histogram_estimator_range`
- Ownership: `opaque_handle`

## `histogram_bin_edges`

Returns inferred fixed-count histogram edges.

- C symbol: `terlan_ndarray_v1_numpy_histogram_bin_edges`
- NativeBoundary operation: `ndarray.numpy.histogram_bin_edges`
- Ownership: `opaque_handle`

## `histogram_bin_edges_estimator`

Returns estimator-selected histogram edges.

- C symbol: `terlan_ndarray_v1_numpy_histogram_bin_edges_estimator`
- NativeBoundary operation: `ndarray.numpy.histogram_bin_edges_estimator`
- Ownership: `opaque_handle`

## `histogram2d`

Builds a two-dimensional histogram with equal bin counts and inferred axis ranges.

- C symbol: `terlan_ndarray_v1_histogram2d`
- NativeBoundary operation: `ndarray.numpy.histogram2d`
- Ownership: `opaque_handle`

## `histogram2d_range`

Builds a two-dimensional histogram with explicit per-axis counts and ranges.

- C symbol: `terlan_ndarray_v1_histogram2d_range`
- NativeBoundary operation: `ndarray.numpy.histogram2d_range`
- Ownership: `opaque_handle`

## `histogram2d_edges`

Builds a two-dimensional histogram from explicit edge vectors.

- C symbol: `terlan_ndarray_v1_histogram2d_edges`
- NativeBoundary operation: `ndarray.numpy.histogram2d_edges`
- Ownership: `opaque_handle`

## `histogram2d_x_edges`

Builds a histogram with explicit x edges and uniform y bins.

- C symbol: `terlan_ndarray_v1_histogram2d_x_edges`
- NativeBoundary operation: `ndarray.numpy.histogram2d_x_edges`
- Ownership: `opaque_handle`

## `histogram2d_y_edges`

Builds a histogram with uniform x bins and explicit y edges.

- C symbol: `terlan_ndarray_v1_histogram2d_y_edges`
- NativeBoundary operation: `ndarray.numpy.histogram2d_y_edges`
- Ownership: `opaque_handle`

## `histogramdd`

Builds a multidimensional histogram with one inferred-range bin count.

- C symbol: `terlan_ndarray_v1_numpy_histogramdd`
- NativeBoundary operation: `ndarray.numpy.histogramdd`
- Ownership: `opaque_handle`

## `histogramdd_weighted`

Builds a weighted multidimensional histogram with one inferred-range bin count.

- C symbol: `terlan_ndarray_v1_numpy_histogramdd_weighted`
- NativeBoundary operation: `ndarray.numpy.histogramdd_weighted`
- Ownership: `opaque_handle`

## `hsplit`

Provides NumPy-compatible hsplit behavior.

- C symbol: `terlan_ndarray_v1_hsplit`
- NativeBoundary operation: `ndarray.numpy.hsplit`
- Ownership: `opaque_handle`

## `hstack`

Provides NumPy-compatible hstack behavior.

- C symbol: `terlan_ndarray_v1_hstack`
- NativeBoundary operation: `ndarray.numpy.hstack`
- Ownership: `opaque_handle`

## `insert`

Provides NumPy-compatible insert behavior.

- C symbol: `terlan_ndarray_v1_insert`
- NativeBoundary operation: `ndarray.numpy.insert`
- Ownership: `opaque_handle`

## `insert`

Inserts values before stable flattened integer-index positions.

- C symbol: `terlan_ndarray_v1_insert_indexes`
- NativeBoundary operation: `ndarray.numpy.insert_indexes`
- Ownership: `opaque_handle`

## `insert`

Inserts values before one scalar index along a normalized axis.

- C symbol: `terlan_ndarray_v1_insert_scalar_axis`
- NativeBoundary operation: `ndarray.numpy.insert_scalar_axis`
- Ownership: `opaque_handle`

## `insert`

Inserts values before stable integer-index positions along a normalized axis.

- C symbol: `terlan_ndarray_v1_insert_indexes_axis`
- NativeBoundary operation: `ndarray.numpy.insert_indexes_axis`
- Ownership: `opaque_handle`

## `ix_`

Provides NumPy-compatible ix  behavior.

- C symbol: `terlan_ndarray_v1_ix`
- NativeBoundary operation: `ndarray.numpy.ix_`
- Ownership: `opaque_handle`

## `may_share_memory`

Provides NumPy-compatible may share memory behavior.

- C symbol: `terlan_ndarray_v1_may_share_memory`
- NativeBoundary operation: `ndarray.numpy.may_share_memory`
- Ownership: `value`

## `may_share_memory_with_work`

Returns an exact overlap answer within a work budget and conservatively returns true when the budget is exhausted; zero selects the bounds-only query and negative one selects unlimited exact work.

- C symbol: `terlan_ndarray_v1_may_share_memory_with_work`
- NativeBoundary operation: `ndarray.numpy.may_share_memory_with_work`
- Ownership: `value`

## `internal_overlap`

Returns whether two or more logical indexes in one strided array address the same element.

- C symbol: `terlan_ndarray_v1_internal_overlap`
- NativeBoundary operation: `ndarray.numpy.internal_overlap`
- Ownership: `value`

## `msort`

Provides NumPy-compatible msort behavior.

- C symbol: `terlan_ndarray_v1_msort`
- NativeBoundary operation: `ndarray.numpy.msort`
- Ownership: `opaque_handle`

## `pad`

Provides NumPy-compatible pad behavior.

- C symbol: `terlan_ndarray_v1_pad`
- NativeBoundary operation: `ndarray.numpy.pad`
- Ownership: `opaque_handle`

## `place`

Provides NumPy-compatible place behavior.

- C symbol: `terlan_ndarray_v1_place`
- NativeBoundary operation: `ndarray.numpy.place`
- Ownership: `value`

## `putmask`

Provides NumPy-compatible putmask behavior.

- C symbol: `terlan_ndarray_v1_putmask`
- NativeBoundary operation: `ndarray.numpy.putmask`
- Ownership: `value`

## `row_stack`

Provides NumPy-compatible row stack behavior.

- C symbol: `terlan_ndarray_v1_row_stack`
- NativeBoundary operation: `ndarray.numpy.row_stack`
- Ownership: `opaque_handle`

## `select`

Provides NumPy-compatible select behavior.

- C symbol: `terlan_ndarray_v1_select`
- NativeBoundary operation: `ndarray.numpy.select`
- Ownership: `opaque_handle`

## `shares_memory`

Provides NumPy-compatible shares memory behavior.

- C symbol: `terlan_ndarray_v1_shares_memory`
- NativeBoundary operation: `ndarray.numpy.shares_memory`
- Ownership: `value`

## `shares_memory_with_work`

Returns exact memory overlap within the supplied work budget, failing when finite work is exhausted; negative one selects unlimited exact work.

- C symbol: `terlan_ndarray_v1_shares_memory_with_work`
- NativeBoundary operation: `ndarray.numpy.shares_memory_with_work`
- Ownership: `value`

## `tril_indices`

Provides NumPy-compatible tril indices behavior.

- C symbol: `terlan_ndarray_v1_tril_indices`
- NativeBoundary operation: `ndarray.numpy.tril_indices`
- Ownership: `opaque_handle`

## `tril_indices_from`

Provides NumPy-compatible tril indices from behavior.

- C symbol: `terlan_ndarray_v1_tril_indices_from`
- NativeBoundary operation: `ndarray.numpy.tril_indices_from`
- Ownership: `opaque_handle`

## `triu_indices`

Provides NumPy-compatible triu indices behavior.

- C symbol: `terlan_ndarray_v1_triu_indices`
- NativeBoundary operation: `ndarray.numpy.triu_indices`
- Ownership: `opaque_handle`

## `triu_indices_from`

Provides NumPy-compatible triu indices from behavior.

- C symbol: `terlan_ndarray_v1_triu_indices_from`
- NativeBoundary operation: `ndarray.numpy.triu_indices_from`
- Ownership: `opaque_handle`

## `vsplit`

Provides NumPy-compatible vsplit behavior.

- C symbol: `terlan_ndarray_v1_vsplit`
- NativeBoundary operation: `ndarray.numpy.vsplit`
- Ownership: `opaque_handle`

## `vstack`

Provides NumPy-compatible vstack behavior.

- C symbol: `terlan_ndarray_v1_vstack`
- NativeBoundary operation: `ndarray.numpy.vstack`
- Ownership: `opaque_handle`

## `average_weighted`

Returns an average weighted by a same-shape array or one-dimensional axis weights.

- C symbol: `terlan_ndarray_v1_weighted_average_axes`
- NativeBoundary operation: `ndarray.numpy.average_weighted`
- Ownership: `opaque_handle`

## `average_with_sum`

Returns an unweighted average followed by its broadcast reduction cardinality.

- C symbol: `terlan_ndarray_v1_average_with_sum`
- NativeBoundary operation: `ndarray.numpy.average_with_sum`
- Ownership: `value`

## `average_weighted_with_sum`

Returns a weighted average followed by its broadcast weight sum.

- C symbol: `terlan_ndarray_v1_weighted_average_with_sum`
- NativeBoundary operation: `ndarray.numpy.average_weighted_with_sum`
- Ownership: `value`

## `covariance_typed`

Returns covariance in the requested floating-point or complex dtype.

- C symbol: `terlan_ndarray_v1_covariance_typed`
- NativeBoundary operation: `ndarray.numpy.covariance_typed`
- Ownership: `opaque_handle`

## `covariance_pair`

Returns covariance after appending variables from a second observation array.

- C symbol: `terlan_ndarray_v1_covariance_pair`
- NativeBoundary operation: `ndarray.numpy.covariance_pair`
- Ownership: `opaque_handle`

## `covariance_frequency_weighted`

Returns covariance with nonnegative integral observation frequencies.

- C symbol: `terlan_ndarray_v1_covariance_frequency_weighted`
- NativeBoundary operation: `ndarray.numpy.covariance_frequency_weighted`
- Ownership: `opaque_handle`

## `covariance_analytic_weighted`

Returns covariance with nonnegative relative observation weights.

- C symbol: `terlan_ndarray_v1_covariance_analytic_weighted`
- NativeBoundary operation: `ndarray.numpy.covariance_analytic_weighted`
- Ownership: `opaque_handle`

## `covariance_weighted`

Returns covariance with combined frequency and analytic observation weights.

- C symbol: `terlan_ndarray_v1_covariance_weighted`
- NativeBoundary operation: `ndarray.numpy.covariance_weighted`
- Ownership: `opaque_handle`

## `covariance_pair_weighted`

Returns paired covariance with combined observation weights.

- C symbol: `terlan_ndarray_v1_covariance_pair_weighted`
- NativeBoundary operation: `ndarray.numpy.covariance_pair_weighted`
- Ownership: `opaque_handle`

## `correlation_typed`

Returns correlation coefficients in the requested floating-point or complex dtype.

- C symbol: `terlan_ndarray_v1_correlation_typed`
- NativeBoundary operation: `ndarray.numpy.correlation_typed`
- Ownership: `opaque_handle`

## `correlation_pair`

Returns correlation coefficients after appending variables from a second observation array.

- C symbol: `terlan_ndarray_v1_correlation_pair`
- NativeBoundary operation: `ndarray.numpy.correlation_pair`
- Ownership: `opaque_handle`

## `nansum`

Returns sums while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_sum`
- NativeBoundary operation: `ndarray.numpy.nansum`
- Ownership: `opaque_handle`

## `nanprod`

Returns products while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_product`
- NativeBoundary operation: `ndarray.numpy.nanprod`
- Ownership: `opaque_handle`

## `nanmean`

Returns means while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_mean`
- NativeBoundary operation: `ndarray.numpy.nanmean`
- Ownership: `opaque_handle`

## `nanmin`

Returns minima while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_minimum`
- NativeBoundary operation: `ndarray.numpy.nanmin`
- Ownership: `opaque_handle`

## `nanmax`

Returns maxima while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_maximum`
- NativeBoundary operation: `ndarray.numpy.nanmax`
- Ownership: `opaque_handle`

## `nanvar`

Returns variance while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_variance`
- NativeBoundary operation: `ndarray.numpy.nanvar`
- Ownership: `opaque_handle`

## `nanstd`

Returns standard deviation while omitting NaN values.

- C symbol: `terlan_ndarray_v1_nan_standard_deviation`
- NativeBoundary operation: `ndarray.numpy.nanstd`
- Ownership: `opaque_handle`

## `nansum_as`

Returns NaN-omitting sums in an explicit dtype.

- C symbol: `terlan_ndarray_v1_nan_sum_as`
- NativeBoundary operation: `ndarray.numpy.nansum_as`
- Ownership: `opaque_handle`

## `nanprod_as`

Returns NaN-omitting products in an explicit dtype.

- C symbol: `terlan_ndarray_v1_nan_product_as`
- NativeBoundary operation: `ndarray.numpy.nanprod_as`
- Ownership: `opaque_handle`

## `nanmean_as`

Returns NaN-omitting means in an explicit inexact dtype.

- C symbol: `terlan_ndarray_v1_nan_mean_as`
- NativeBoundary operation: `ndarray.numpy.nanmean_as`
- Ownership: `opaque_handle`

## `nanvar_as`

Returns NaN-omitting variances in an explicit inexact dtype.

- C symbol: `terlan_ndarray_v1_nan_variance_as`
- NativeBoundary operation: `ndarray.numpy.nanvar_as`
- Ownership: `opaque_handle`

## `nanstd_as`

Returns NaN-omitting standard deviations in an explicit inexact dtype.

- C symbol: `terlan_ndarray_v1_nan_standard_deviation_as`
- NativeBoundary operation: `ndarray.numpy.nanstd_as`
- Ownership: `opaque_handle`

## `nanargmin_axes`

Returns minimum indexes over normalized axes while omitting NaNs.

- C symbol: `terlan_ndarray_v1_nan_argmin`
- NativeBoundary operation: `ndarray.numpy.nanargmin_axes`
- Ownership: `opaque_handle`

## `nanargmax_axes`

Returns maximum indexes over normalized axes while omitting NaNs.

- C symbol: `terlan_ndarray_v1_nan_argmax`
- NativeBoundary operation: `ndarray.numpy.nanargmax_axes`
- Ownership: `opaque_handle`

## `nansum_into`

Writes NaN-omitting sums into an exact-shape destination.

- C symbol: `terlan_ndarray_v1_nan_sum_into`
- NativeBoundary operation: `ndarray.numpy.nansum_into`
- Ownership: `value`

## `nanprod_into`

Writes NaN-omitting products into an exact-shape destination.

- C symbol: `terlan_ndarray_v1_nan_product_into`
- NativeBoundary operation: `ndarray.numpy.nanprod_into`
- Ownership: `value`

## `nanmean_into`

Writes NaN-omitting means into an exact-shape inexact destination.

- C symbol: `terlan_ndarray_v1_nan_mean_into`
- NativeBoundary operation: `ndarray.numpy.nanmean_into`
- Ownership: `value`

## `nanmin_into`

Writes NaN-omitting minima into an exact-shape destination.

- C symbol: `terlan_ndarray_v1_nan_minimum_into`
- NativeBoundary operation: `ndarray.numpy.nanmin_into`
- Ownership: `value`

## `nanmax_into`

Writes NaN-omitting maxima into an exact-shape destination.

- C symbol: `terlan_ndarray_v1_nan_maximum_into`
- NativeBoundary operation: `ndarray.numpy.nanmax_into`
- Ownership: `value`

## `nanvar_into`

Writes NaN-omitting variances into an exact-shape inexact destination.

- C symbol: `terlan_ndarray_v1_nan_variance_into`
- NativeBoundary operation: `ndarray.numpy.nanvar_into`
- Ownership: `value`

## `nanstd_into`

Writes NaN-omitting standard deviations into an exact-shape inexact destination.

- C symbol: `terlan_ndarray_v1_nan_standard_deviation_into`
- NativeBoundary operation: `ndarray.numpy.nanstd_into`
- Ownership: `value`

## `nancumsum_flat`

Returns a flattened cumulative sum with NaNs replaced by zero.

- C symbol: `terlan_ndarray_v1_nancumsum_flat`
- NativeBoundary operation: `ndarray.numpy.nancumsum_flat`
- Ownership: `opaque_handle`

## `nancumprod_flat`

Returns a flattened cumulative product with NaNs replaced by one.

- C symbol: `terlan_ndarray_v1_nancumprod_flat`
- NativeBoundary operation: `ndarray.numpy.nancumprod_flat`
- Ownership: `opaque_handle`

## `nancumsum_as`

Returns a NaN-normalized cumulative sum in an explicit dtype.

- C symbol: `terlan_ndarray_v1_nancumsum_as`
- NativeBoundary operation: `ndarray.numpy.nancumsum_as`
- Ownership: `opaque_handle`

## `nancumprod_as`

Returns a NaN-normalized cumulative product in an explicit dtype.

- C symbol: `terlan_ndarray_v1_nancumprod_as`
- NativeBoundary operation: `ndarray.numpy.nancumprod_as`
- Ownership: `opaque_handle`

## `nancumsum_into`

Writes a NaN-normalized cumulative sum into an exact-shape destination.

- C symbol: `terlan_ndarray_v1_nancumsum_into`
- NativeBoundary operation: `ndarray.numpy.nancumsum_into`
- Ownership: `value`

## `nancumprod_into`

Writes a NaN-normalized cumulative product into an exact-shape destination.

- C symbol: `terlan_ndarray_v1_nancumprod_into`
- NativeBoundary operation: `ndarray.numpy.nancumprod_into`
- Ownership: `value`

## `nancumsum_flat_into`

Writes a flattened NaN-normalized cumulative sum into a vector destination.

- C symbol: `terlan_ndarray_v1_nancumsum_flat_into`
- NativeBoundary operation: `ndarray.numpy.nancumsum_flat_into`
- Ownership: `value`

## `nancumprod_flat_into`

Writes a flattened NaN-normalized cumulative product into a vector destination.

- C symbol: `terlan_ndarray_v1_nancumprod_flat_into`
- NativeBoundary operation: `ndarray.numpy.nancumprod_flat_into`
- Ownership: `value`

## `nan_to_num_default`

Replaces NaN and infinity components with dtype-specific finite defaults.

- C symbol: `terlan_ndarray_v1_nan_to_num_default`
- NativeBoundary operation: `ndarray.numpy.nan_to_num_default`
- Ownership: `opaque_handle`

## `nan_to_num_into`

Writes non-finite replacements into an exact-shape, exact-dtype destination.

- C symbol: `terlan_ndarray_v1_nan_to_num_into`
- NativeBoundary operation: `ndarray.numpy.nan_to_num_into`
- Ownership: `opaque_handle`

## `add_into`

Writes a broadcast addition into an exact-shape destination after staging all aliased inputs.

- C symbol: `terlan_ndarray_v1_add_into`
- NativeBoundary operation: `ndarray.numpy.add_into`
- Ownership: `value`

## `add_where_into`

Writes broadcast sums where a Bool mask is true while preserving masked destination elements and staging aliases.

- C symbol: `terlan_ndarray_v1_add_where_into`
- NativeBoundary operation: `ndarray.numpy.add_where_into`
- Ownership: `value`

## `absolute_into`

Writes magnitudes into an exact-shape destination after staging aliased input.

- C symbol: `terlan_ndarray_v1_absolute_into`
- NativeBoundary operation: `ndarray.numpy.absolute_into`
- Ownership: `value`

## `logical_not_into`

Writes truth-value negation into an exact-shape Bool destination.

- C symbol: `terlan_ndarray_v1_logical_not_into`
- NativeBoundary operation: `ndarray.numpy.logical_not_into`
- Ownership: `value`

## `logical_and_into`

Writes broadcast truth-value conjunction into an exact-shape Bool destination.

- C symbol: `terlan_ndarray_v1_logical_and_into`
- NativeBoundary operation: `ndarray.numpy.logical_and_into`
- Ownership: `value`

## `logical_or_into`

Writes broadcast truth-value disjunction into an exact-shape Bool destination.

- C symbol: `terlan_ndarray_v1_logical_or_into`
- NativeBoundary operation: `ndarray.numpy.logical_or_into`
- Ownership: `value`

## `logical_xor_into`

Writes broadcast truth-value exclusive disjunction into an exact-shape Bool destination.

- C symbol: `terlan_ndarray_v1_logical_xor_into`
- NativeBoundary operation: `ndarray.numpy.logical_xor_into`
- Ownership: `value`

## `invert_into`

Writes bitwise inversion into an exact-shape destination after staging aliased input.

- C symbol: `terlan_ndarray_v1_invert_into`
- NativeBoundary operation: `ndarray.numpy.invert_into`
- Ownership: `value`

## `invert_where_into`

Writes bitwise inversion where a Bool mask is true and stages masks that alias the destination.

- C symbol: `terlan_ndarray_v1_invert_where_into`
- NativeBoundary operation: `ndarray.numpy.invert_where_into`
- Ownership: `value`

## `reciprocal_into`

Writes reciprocals into an exact-shape destination after staging aliased input.

- C symbol: `terlan_ndarray_v1_reciprocal_into`
- NativeBoundary operation: `ndarray.numpy.reciprocal_into`
- Ownership: `value`

## `reciprocal_where_into`

Writes reciprocals where a Bool mask is true while preserving masked destination elements.

- C symbol: `terlan_ndarray_v1_reciprocal_where_into`
- NativeBoundary operation: `ndarray.numpy.reciprocal_where_into`
- Ownership: `value`

## `invert_at`

Applies unbuffered bitwise inversion at rank-one integer indexes after stabilizing aliased indexes.

- C symbol: `terlan_ndarray_v1_invert_at`
- NativeBoundary operation: `ndarray.numpy.invert_at`
- Ownership: `value`

## `add_at`

Applies sequential unbuffered addition at rank-one integer indexes with broadcast values and stable aliases.

- C symbol: `terlan_ndarray_v1_add_at`
- NativeBoundary operation: `ndarray.numpy.add_at`
- Ownership: `value`

## `logical_xor_reduce`

Returns parity reductions of truthy values over selected axes.

- C symbol: `terlan_ndarray_v1_logical_xor_reduce`
- NativeBoundary operation: `ndarray.numpy.logical_xor_reduce`
- Ownership: `opaque_handle`

## `add_reduceat`

Returns segmented addition reductions beginning at each rank-one integer index along an axis.

- C symbol: `terlan_ndarray_v1_add_reduceat`
- NativeBoundary operation: `ndarray.numpy.add_reduceat`
- Ownership: `opaque_handle`

## `add_reduceat_into`

Writes segmented addition reductions into an exact-shape destination after staging aliased inputs.

- C symbol: `terlan_ndarray_v1_add_reduceat_into`
- NativeBoundary operation: `ndarray.numpy.add_reduceat_into`
- Ownership: `value`
