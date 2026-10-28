#ifndef TERLAN_NDARRAY_H
#define TERLAN_NDARRAY_H
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t TerlanNdArrayStatus;
typedef struct TerlanNdArray TerlanNdArray;
typedef TerlanNdArray *TerlanNdArrayHandle;
typedef struct TerlanNdRandomStream TerlanNdRandomStream;
typedef TerlanNdRandomStream *TerlanNdRandomStreamHandle;
typedef struct TerlanNdArrowArray TerlanNdArrowArray;
typedef TerlanNdArrowArray *TerlanNdArrowArrayHandle;

enum {
  TERLAN_NDARRAY_OK = 0,
  TERLAN_NDARRAY_INVALID_ARGUMENT = 100,
  TERLAN_NDARRAY_NEGATIVE_DIMENSION = 101,
  TERLAN_NDARRAY_ELEMENT_COUNT_MISMATCH = 102,
  TERLAN_NDARRAY_SIZE_OVERFLOW = 103,
  TERLAN_NDARRAY_ALLOCATION_FAILED = 104,
  TERLAN_NDARRAY_INVALID_BOOL = 105,
  TERLAN_NDARRAY_DTYPE_MISMATCH = 106,
  TERLAN_NDARRAY_DISPOSED = 107,
  TERLAN_NDARRAY_SHAPE_MISMATCH = 108,
  TERLAN_NDARRAY_UNSUPPORTED_DTYPE = 109,
  TERLAN_NDARRAY_AXIS_OUT_OF_RANGE = 110,
  TERLAN_NDARRAY_DUPLICATE_AXIS = 111,
  TERLAN_NDARRAY_ARITHMETIC_OVERFLOW = 112,
  TERLAN_NDARRAY_DLPACK_VERSION = 113,
  TERLAN_NDARRAY_DLPACK_DEVICE = 114,
  TERLAN_NDARRAY_DLPACK_DTYPE = 115,
  TERLAN_NDARRAY_DLPACK_LAYOUT = 116,
  TERLAN_NDARRAY_DLPACK_DELETER = 117,
  TERLAN_NDARRAY_BLAS_MISSING = 118,
  TERLAN_NDARRAY_BLAS_INTEGER_WIDTH = 119,
  TERLAN_NDARRAY_INDEX_OUT_OF_RANGE = 120,
  TERLAN_NDARRAY_READ_ONLY = 121,
  TERLAN_NDARRAY_INVALID_SLICE = 122,
  TERLAN_NDARRAY_UNSUPPORTED_LAYOUT = 123,
  TERLAN_NDARRAY_CAST_OVERFLOW = 124,
  TERLAN_NDARRAY_DIVIDE_BY_ZERO = 125,
  TERLAN_NDARRAY_DOMAIN_ERROR = 126,
  TERLAN_NDARRAY_MALFORMED_DATA = 127,
  TERLAN_NDARRAY_INTEGRITY_ERROR = 128,
  TERLAN_NDARRAY_IO_ERROR = 129,
  TERLAN_NDARRAY_TOO_HARD = 130
};

enum {
  TERLAN_NDARRAY_DTYPE_BOOL = 1,
  TERLAN_NDARRAY_DTYPE_INT64 = 2,
  TERLAN_NDARRAY_DTYPE_FLOAT64 = 3,
  TERLAN_NDARRAY_DTYPE_INT8 = 4,
  TERLAN_NDARRAY_DTYPE_INT16 = 5,
  TERLAN_NDARRAY_DTYPE_INT32 = 6,
  TERLAN_NDARRAY_DTYPE_UINT8 = 7,
  TERLAN_NDARRAY_DTYPE_UINT16 = 8,
  TERLAN_NDARRAY_DTYPE_UINT32 = 9,
  TERLAN_NDARRAY_DTYPE_UINT64 = 10,
  TERLAN_NDARRAY_DTYPE_FLOAT32 = 11,
  TERLAN_NDARRAY_DTYPE_COMPLEX64 = 12,
  TERLAN_NDARRAY_DTYPE_COMPLEX128 = 13,
  TERLAN_NDARRAY_DTYPE_FLOAT16 = 14
};

enum { TERLAN_NDARRAY_DEVICE_CPU = 1 };

TerlanNdArrayStatus terlan_ndarray_v1_from_bools( const uint8_t *values, int64_t value_count, const int64_t *shape, int64_t rank, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_from_ints( const int64_t *values, int64_t value_count, const int64_t *shape, int64_t rank, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_from_floats( const double *values, int64_t value_count, const int64_t *shape, int64_t rank, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_from_ints_as( const int64_t *values, int64_t value_count, const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_from_floats_as( const double *values, int64_t value_count, const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_from_complex( const double *real_values, int64_t real_count, const double *imag_values, int64_t imag_count, const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_zeros( const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_empty( const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_ones( const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_full_int( const int64_t *shape, int64_t rank, int64_t value, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_full_float( const int64_t *shape, int64_t rank, double value, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arange( double start, double stop, double step, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arange_float_stop(double stop, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arange_float_interval(double start, double stop, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arange_float_step(double start, double stop, double step, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arange_int(int64_t start, int64_t stop, int64_t step, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arange_int_stop(int64_t stop, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arange_int_interval(int64_t start, int64_t stop, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_linspace( double start, double stop, int64_t count, int32_t endpoint, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_linspace_arrays(TerlanNdArrayHandle start, TerlanNdArrayHandle stop, int64_t count, int32_t endpoint, int64_t axis, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_linspace_steps(TerlanNdArrayHandle start, TerlanNdArrayHandle stop, int64_t count, int32_t endpoint, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_identity( int64_t rows, int64_t columns, int64_t diagonal, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_eye(int64_t rows, int64_t columns, int64_t diagonal, int32_t dtype, int32_t order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_empty_like(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_zeros_like(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_ones_like(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_full_like_int( TerlanNdArrayHandle array, int64_t value, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_full_like_float( TerlanNdArrayHandle array, double value, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_logspace( double start, double stop, int64_t count, int32_t endpoint, double base, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_geomspace( double start, double stop, int64_t count, int32_t endpoint, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_logspace_arrays(TerlanNdArrayHandle start, TerlanNdArrayHandle stop, int64_t count, int32_t endpoint, int64_t axis, double base, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_geomspace_arrays(TerlanNdArrayHandle start, TerlanNdArrayHandle stop, int64_t count, int32_t endpoint, int64_t axis, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_tri( int64_t rows, int64_t columns, int64_t diagonal, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_tril( TerlanNdArrayHandle array, int64_t diagonal, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_triu( TerlanNdArrayHandle array, int64_t diagonal, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_diagflat( TerlanNdArrayHandle array, int64_t diagonal, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_indices( const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_vander( TerlanNdArrayHandle array, int64_t columns, int32_t increasing, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_choose( TerlanNdArrayHandle indexes, const uint64_t *choices, int64_t choice_count, int32_t mode, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_choose_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle indexes, const uint64_t *choices, int64_t choice_count, int32_t mode);
TerlanNdArrayStatus terlan_ndarray_v1_fill( TerlanNdArrayHandle array, TerlanNdArrayHandle value);
TerlanNdArrayStatus terlan_ndarray_v1_resize( TerlanNdArrayHandle array, const int64_t *shape, int64_t rank, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_uniform( const int64_t *shape, int64_t rank, double low, double high, int64_t seed, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_normal( const int64_t *shape, int64_t rank, double mean, double standard_deviation, int64_t seed, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_integers( const int64_t *shape, int64_t rank, int64_t low, int64_t high, int64_t seed, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_bernoulli( const int64_t *shape, int64_t rank, double probability, int64_t seed, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_permutation(int64_t length, int64_t seed, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_shuffle( TerlanNdArrayHandle array, int64_t axis, int64_t seed, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_new(int64_t seed, TerlanNdRandomStreamHandle *out_stream);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_beta( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double alpha, double beta, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_gamma( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double distribution_shape, double scale, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_binomial( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, int64_t trials, double probability, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_poisson( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double lambda, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_standard_normal( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_chisquare( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double degrees_of_freedom, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_f( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double numerator_degrees, double denominator_degrees, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_gumbel( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double location, double scale, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_lognormal( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double mean, double sigma, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_pareto( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double distribution_shape, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_exponential( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double scale, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_standard_exponential( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_standard_gamma( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double distribution_shape, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_standard_t( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double degrees_of_freedom, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_triangular( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double left, double mode, double right, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_weibull( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double distribution_shape, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_geometric( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double probability, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_hypergeometric( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, int64_t good, int64_t bad, int64_t samples, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_zipf( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double exponent, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_standard_cauchy( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_laplace( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double location, double scale, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_logistic( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double location, double scale, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_negative_binomial( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double successes, double probability, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_noncentral_chisquare( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double degrees_of_freedom, double noncentrality, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_noncentral_f( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double numerator_degrees, double denominator_degrees, double noncentrality, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_rayleigh( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double scale, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_wald( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double mean, double scale, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_logseries( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double probability, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_vonmises( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double mean, double concentration, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_choice(TerlanNdRandomStreamHandle stream, TerlanNdArrayHandle source, const int64_t *shape, int64_t rank, TerlanNdArrayHandle probabilities, int32_t replacement, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_dirichlet(TerlanNdRandomStreamHandle stream, TerlanNdArrayHandle alpha, const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_multinomial(TerlanNdRandomStreamHandle stream, int64_t trials, TerlanNdArrayHandle probabilities, const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_multivariate_hypergeometric(TerlanNdRandomStreamHandle stream, TerlanNdArrayHandle colors, int64_t samples, const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_multivariate_normal(TerlanNdRandomStreamHandle stream, TerlanNdArrayHandle mean, TerlanNdArrayHandle covariance, const int64_t *shape, int64_t rank, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_delete(TerlanNdRandomStreamHandle stream);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_uniform( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double low, double high, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_normal( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double mean, double standard_deviation, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_integers( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, int64_t low, int64_t high, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_bernoulli( TerlanNdRandomStreamHandle stream, const int64_t *shape, int64_t rank, double probability, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_permutation( TerlanNdRandomStreamHandle stream, int64_t length, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_random_stream_shuffle( TerlanNdRandomStreamHandle stream, TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_determinant(TerlanNdArrayHandle array, double *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_solve(TerlanNdArrayHandle matrix, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_inverse(TerlanNdArrayHandle matrix, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_norm_l1(TerlanNdArrayHandle array, double *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_norm_l2(TerlanNdArrayHandle array, double *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_norm_max(TerlanNdArrayHandle array, double *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_frobenius_norm(TerlanNdArrayHandle array, double *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_nuclear_norm(TerlanNdArrayHandle array, double *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_matrix_rank(TerlanNdArrayHandle array, double tolerance, int64_t *out_rank);
TerlanNdArrayStatus terlan_ndarray_v1_least_squares(TerlanNdArrayHandle matrix, TerlanNdArrayHandle rhs, double tolerance, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_cholesky(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_qr(TerlanNdArrayHandle array, int32_t reduced, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_svd(TerlanNdArrayHandle array, int32_t full_matrices, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_symmetric_eigen(TerlanNdArrayHandle array, double tolerance, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_cond(TerlanNdArrayHandle array, int64_t order, double *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_eig(TerlanNdArrayHandle array, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_eigvals(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_eigvalsh(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_matrix_power(TerlanNdArrayHandle array, int64_t exponent, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_multi_dot(const uint64_t *arrays, int64_t array_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_norm(TerlanNdArrayHandle array, double *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_pinv(TerlanNdArrayHandle array, double tolerance, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_slogdet(TerlanNdArrayHandle array, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_tensorinv(TerlanNdArrayHandle array, int64_t index, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_tensorsolve(TerlanNdArrayHandle coefficients, TerlanNdArrayHandle rhs, const int64_t *axes, int64_t axis_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_to_npy( TerlanNdArrayHandle array, uint8_t **out_bytes, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_from_npy( const uint8_t *bytes, int64_t length, int64_t maximum, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_serialize( TerlanNdArrayHandle array, uint8_t **out_bytes, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_deserialize( const uint8_t *bytes, int64_t length, int64_t maximum, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_mmap_npy( const char *path, int32_t writable, int64_t maximum, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_flush(TerlanNdArrayHandle array);
TerlanNdArrayStatus terlan_ndarray_v1_to_arrow( TerlanNdArrayHandle array, TerlanNdArrowArrayHandle *out_arrow);
TerlanNdArrayStatus terlan_ndarray_v1_from_arrow( TerlanNdArrowArrayHandle arrow, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_import_arrow_c( void *array, void *schema, const int64_t *shape, int64_t rank, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arrow_length( TerlanNdArrowArrayHandle arrow, int64_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_arrow_null_count( TerlanNdArrowArrayHandle arrow, int64_t *out_null_count);
const void *terlan_ndarray_v1_arrow_array_pointer(
    TerlanNdArrowArrayHandle arrow);
const void *terlan_ndarray_v1_arrow_schema_pointer(
    TerlanNdArrowArrayHandle arrow);
TerlanNdArrayStatus terlan_ndarray_v1_arrow_delete( TerlanNdArrowArrayHandle arrow);
TerlanNdArrayStatus terlan_ndarray_v1_diagonal( TerlanNdArrayHandle array, int64_t offset, int64_t axis_a, int64_t axis_b, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_astype( TerlanNdArrayHandle array, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_astype_ordered(
    TerlanNdArrayHandle array, int32_t dtype, int32_t order, int32_t copy,
    int32_t casting, TerlanNdArrayHandle *out_array);

TerlanNdArrayStatus terlan_ndarray_v1_rank(TerlanNdArrayHandle array, int64_t *out_rank);
TerlanNdArrayStatus terlan_ndarray_v1_shape(TerlanNdArrayHandle array, int64_t **out_shape);
TerlanNdArrayStatus terlan_ndarray_v1_strides(TerlanNdArrayHandle array, int64_t **out_strides);
TerlanNdArrayStatus terlan_ndarray_v1_numel(TerlanNdArrayHandle array, int64_t *out_numel);
TerlanNdArrayStatus terlan_ndarray_v1_byte_count(TerlanNdArrayHandle array, int64_t *out_byte_count);
TerlanNdArrayStatus terlan_ndarray_v1_dtype(TerlanNdArrayHandle array, int32_t *out_dtype);
TerlanNdArrayStatus terlan_ndarray_v1_is_contiguous( TerlanNdArrayHandle array, int32_t *out_contiguous);
TerlanNdArrayStatus terlan_ndarray_v1_is_fortran_contiguous(
    TerlanNdArrayHandle array, int32_t *out_contiguous);
TerlanNdArrayStatus terlan_ndarray_v1_device_type( TerlanNdArrayHandle array, int32_t *out_device_type);
TerlanNdArrayStatus terlan_ndarray_v1_to_bools( TerlanNdArrayHandle array, uint8_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_to_ints( TerlanNdArrayHandle array, int64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_to_floats( TerlanNdArrayHandle array, double **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_to_ints_cast( TerlanNdArrayHandle array, int64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_to_floats_cast( TerlanNdArrayHandle array, double **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_complex_reals( TerlanNdArrayHandle array, double **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_complex_imags( TerlanNdArrayHandle array, double **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_reshape( TerlanNdArrayHandle array, const int64_t *shape, int64_t rank, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_reshape_ordered(TerlanNdArrayHandle array, const int64_t *shape, int64_t rank, int32_t order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_resize_in_place(TerlanNdArrayHandle array, const int64_t *shape, int64_t rank);
TerlanNdArrayStatus terlan_ndarray_v1_transpose( TerlanNdArrayHandle array, int64_t axis_a, int64_t axis_b, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_copy( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_copy_ordered(
    TerlanNdArrayHandle array, int32_t order,
    TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_contiguous( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_index_axis( TerlanNdArrayHandle array, int64_t axis, int64_t index, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_slice_axis( TerlanNdArrayHandle array, int64_t axis, int64_t start, int64_t stop, int64_t step, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_permute( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_take( TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_take_mode( TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count, int64_t axis, int32_t mode, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_take_indices( TerlanNdArrayHandle array, TerlanNdArrayHandle indices, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_take_indices_mode( TerlanNdArrayHandle array, TerlanNdArrayHandle indices, int64_t axis, int32_t mode, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_take_indices_into( TerlanNdArrayHandle destination, TerlanNdArrayHandle array, TerlanNdArrayHandle indices, int64_t axis, int32_t mode);
TerlanNdArrayStatus terlan_ndarray_v1_take_along_axis( TerlanNdArrayHandle array, TerlanNdArrayHandle indices, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_put( TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count, TerlanNdArrayHandle values);
TerlanNdArrayStatus terlan_ndarray_v1_put_along_axis( TerlanNdArrayHandle array, TerlanNdArrayHandle indices, TerlanNdArrayHandle values, int64_t axis);
TerlanNdArrayStatus terlan_ndarray_v1_compress( TerlanNdArrayHandle array, TerlanNdArrayHandle mask, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_compress_flat( TerlanNdArrayHandle array, TerlanNdArrayHandle mask, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_compress_into( TerlanNdArrayHandle destination, TerlanNdArrayHandle array, TerlanNdArrayHandle mask, int64_t axis);
TerlanNdArrayStatus terlan_ndarray_v1_compress_flat_into( TerlanNdArrayHandle destination, TerlanNdArrayHandle array, TerlanNdArrayHandle mask);
TerlanNdArrayStatus terlan_ndarray_v1_mask_select( TerlanNdArrayHandle array, TerlanNdArrayHandle mask, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_mask_assign( TerlanNdArrayHandle array, TerlanNdArrayHandle mask, TerlanNdArrayHandle values);
TerlanNdArrayStatus terlan_ndarray_v1_gather_coordinates( TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_put_coordinates( TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates, TerlanNdArrayHandle values);
TerlanNdArrayStatus terlan_ndarray_v1_add_at_coordinates( TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates, TerlanNdArrayHandle values);
TerlanNdArrayStatus terlan_ndarray_v1_expand_dims( TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_squeeze( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_flip( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_flip_all( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_atleast_1d( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_atleast_2d( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_atleast_3d( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_ravel( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_ravel_ordered(TerlanNdArrayHandle array, int32_t order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_flatten( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_flatten_ordered(TerlanNdArrayHandle array, int32_t order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_moveaxis( TerlanNdArrayHandle array, const int64_t *sources, int64_t source_count, const int64_t *destinations, int64_t destination_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_rollaxis( TerlanNdArrayHandle array, int64_t axis, int64_t start, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_flipud( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_fliplr( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_rot90( TerlanNdArrayHandle array, int64_t rotations, int64_t axis_a, int64_t axis_b, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_roll( TerlanNdArrayHandle array, int64_t shift, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_repeat( TerlanNdArrayHandle array, int64_t repeats, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_repeat_counts( TerlanNdArrayHandle array, const int64_t *repeats, int64_t repeat_count, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_repeat_flat( TerlanNdArrayHandle array, const int64_t *repeats, int64_t repeat_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_tile( TerlanNdArrayHandle array, const int64_t *repetitions, int64_t repetition_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_basic_slice( TerlanNdArrayHandle array, const int64_t *starts, int64_t start_count, const int64_t *stops, int64_t stop_count, const int64_t *steps, int64_t step_count, int64_t ellipsis_index, const int64_t *squeeze_axes, int64_t squeeze_count, const int64_t *new_axes, int64_t new_axis_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_concatenate( const uint64_t *arrays, int64_t array_count, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_stack( const uint64_t *arrays, int64_t array_count, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_constant( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, const double *before_values, int64_t before_value_count, const double *after_values, int64_t after_value_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_constant_values( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, TerlanNdArrayHandle before_values, TerlanNdArrayHandle after_values, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_edge( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_reflect( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_symmetric( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_wrap( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_reflect_odd( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_symmetric_odd( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_linear_ramp( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, const double *before_values, int64_t before_value_count, const double *after_values, int64_t after_value_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_linear_ramp_values( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, TerlanNdArrayHandle before_values, TerlanNdArrayHandle after_values, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_empty( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_statistical( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, const int64_t *before_lengths, int64_t before_length_count, const int64_t *after_lengths, int64_t after_length_count, int64_t mode, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_minimum( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, const int64_t *before_lengths, int64_t before_length_count, const int64_t *after_lengths, int64_t after_length_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_maximum( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, const int64_t *before_lengths, int64_t before_length_count, const int64_t *after_lengths, int64_t after_length_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_mean( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, const int64_t *before_lengths, int64_t before_length_count, const int64_t *after_lengths, int64_t after_length_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad_median( TerlanNdArrayHandle array, const int64_t *before, int64_t before_count, const int64_t *after, int64_t after_count, const int64_t *before_lengths, int64_t before_length_count, const int64_t *after_lengths, int64_t after_length_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sort( TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_argsort( TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sort_unstable( TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_argsort_unstable( TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sort_flat(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_argsort_flat(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sort_inplace(TerlanNdArrayHandle array, int64_t axis);
TerlanNdArrayStatus terlan_ndarray_v1_partition( TerlanNdArrayHandle array, int64_t kth, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_argpartition( TerlanNdArrayHandle array, int64_t kth, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_partition_kths(TerlanNdArrayHandle array, TerlanNdArrayHandle kths, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_argpartition_kths(TerlanNdArrayHandle array, TerlanNdArrayHandle kths, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_partition_flat(TerlanNdArrayHandle array, int64_t kth, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_argpartition_flat(TerlanNdArrayHandle array, int64_t kth, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_partition_kths_flat(TerlanNdArrayHandle array, TerlanNdArrayHandle kths, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_argpartition_kths_flat(TerlanNdArrayHandle array, TerlanNdArrayHandle kths, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_partition_inplace(TerlanNdArrayHandle array, int64_t kth, int64_t axis);
TerlanNdArrayStatus terlan_ndarray_v1_partition_kths_inplace(TerlanNdArrayHandle array, TerlanNdArrayHandle kths, int64_t axis);
TerlanNdArrayStatus terlan_ndarray_v1_searchsorted( TerlanNdArrayHandle sorted, TerlanNdArrayHandle values, int32_t right, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_searchsorted_with_sorter( TerlanNdArrayHandle array, TerlanNdArrayHandle values, TerlanNdArrayHandle sorter, int32_t right, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_lexsort_keys( TerlanNdArrayHandle keys, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_lexsort_axis( TerlanNdArrayHandle keys, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_unique_all( TerlanNdArrayHandle array, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_union( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_intersect( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_set_difference( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_set_xor( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_is_in( TerlanNdArrayHandle elements, TerlanNdArrayHandle test_elements, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_in1d( TerlanNdArrayHandle elements, TerlanNdArrayHandle test_elements, int32_t invert, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_set_difference_unique( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_intersect_all( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_unique_axis_all( TerlanNdArrayHandle array, int64_t axis, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_ediff1d_edges( TerlanNdArrayHandle array, TerlanNdArrayHandle begin, TerlanNdArrayHandle end, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_broadcast_to( TerlanNdArrayHandle array, const int64_t *shape, int64_t rank, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_item_bool(TerlanNdArrayHandle array, bool *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_item_int(TerlanNdArrayHandle array, int64_t *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_get_bool( TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count, bool *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_get_int( TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count, int64_t *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_get_float( TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count, double *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_set_bool( TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count, bool value);
TerlanNdArrayStatus terlan_ndarray_v1_set_int( TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count, int64_t value);
TerlanNdArrayStatus terlan_ndarray_v1_set_float( TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count, double value);
TerlanNdArrayStatus terlan_ndarray_v1_add( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_subtract( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_multiply( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_true_divide( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_floor_divide( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_remainder( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_power( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_elementwise_minimum( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_elementwise_maximum( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_fmin( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_fmax( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_clip( TerlanNdArrayHandle array, TerlanNdArrayHandle minimum, TerlanNdArrayHandle maximum, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_clip_lower(TerlanNdArrayHandle array, TerlanNdArrayHandle minimum, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_clip_upper(TerlanNdArrayHandle array, TerlanNdArrayHandle maximum, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_clip_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, TerlanNdArrayHandle minimum, TerlanNdArrayHandle maximum);
TerlanNdArrayStatus terlan_ndarray_v1_clip_lower_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, TerlanNdArrayHandle minimum);
TerlanNdArrayStatus terlan_ndarray_v1_clip_upper_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, TerlanNdArrayHandle maximum);
TerlanNdArrayStatus terlan_ndarray_v1_clip_inplace(TerlanNdArrayHandle array, TerlanNdArrayHandle minimum, TerlanNdArrayHandle maximum);
TerlanNdArrayStatus terlan_ndarray_v1_clip_lower_inplace(TerlanNdArrayHandle array, TerlanNdArrayHandle minimum);
TerlanNdArrayStatus terlan_ndarray_v1_clip_upper_inplace(TerlanNdArrayHandle array, TerlanNdArrayHandle maximum);
TerlanNdArrayStatus terlan_ndarray_v1_equal( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_not_equal( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_less( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_less_equal( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_greater( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_greater_equal( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_logical_and( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_logical_or( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_logical_xor( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_bitwise_and( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_bitwise_or( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_bitwise_xor( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_left_shift( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_right_shift( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_gcd( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_lcm( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_hypot( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arctan2( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_logaddexp( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_logaddexp2( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_copysign( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nextafter( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_is_close( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, double relative_tolerance, double absolute_tolerance, int32_t equal_nan, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_where( TerlanNdArrayHandle condition, TerlanNdArrayHandle when_true, TerlanNdArrayHandle when_false, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_negative( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_positive(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_absolute( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sign( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_square( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sqrt( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_reciprocal( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_exp( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_expm1( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_log( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_log1p( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_log2( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_log10( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sin( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_cos( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_tan( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sinh( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_cosh( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_tanh( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arcsin( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arccos( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arctan( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arcsinh( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arccosh( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_arctanh( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_floor( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_ceil( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_round( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_round_decimals(TerlanNdArrayHandle array, int64_t decimals, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_round_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, int64_t decimals);
TerlanNdArrayStatus terlan_ndarray_v1_logical_not( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_is_nan( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_is_infinite( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_is_finite( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_conjugate( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_conjugate_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array);
TerlanNdArrayStatus terlan_ndarray_v1_exp2( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_cbrt( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_degrees( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_radians( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sign_bit( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_frexp_all( TerlanNdArrayHandle array, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_modf_all( TerlanNdArrayHandle array, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_sum( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_product( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_mean( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_mean_where(TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_mean_as(TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_mean_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims);
TerlanNdArrayStatus terlan_ndarray_v1_minimum( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_maximum( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_any( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_all( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_logical_xor_reduce(TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_any_where(TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_all_where(TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_argmin( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_argmin_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims);
TerlanNdArrayStatus terlan_ndarray_v1_argmax( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_argmax_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims);
TerlanNdArrayStatus terlan_ndarray_v1_variance( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int64_t correction, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_standard_deviation( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int64_t correction, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_variance_where(TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int64_t correction, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_variance_as(TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int64_t correction, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_variance_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int64_t correction);
TerlanNdArrayStatus terlan_ndarray_v1_standard_deviation_where(TerlanNdArrayHandle array, TerlanNdArrayHandle mask, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int64_t correction, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_standard_deviation_as(TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int64_t correction, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_standard_deviation_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int64_t correction);
TerlanNdArrayStatus terlan_ndarray_v1_median( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_quantile( TerlanNdArrayHandle array, double quantile, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_percentile( TerlanNdArrayHandle array, double percentile, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_quantiles(TerlanNdArrayHandle array, const double *quantiles, int64_t quantile_count, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_percentiles(TerlanNdArrayHandle array, const double *percentiles, int64_t percentile_count, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nanquantile_axes(TerlanNdArrayHandle array, double quantile, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nanpercentile_axes(TerlanNdArrayHandle array, double percentile, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nanquantiles(TerlanNdArrayHandle array, const double *quantiles, int64_t quantile_count, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nanpercentiles(TerlanNdArrayHandle array, const double *percentiles, int64_t percentile_count, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nanmedian_axes(TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_median_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims);
TerlanNdArrayStatus terlan_ndarray_v1_nanmedian_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims);
TerlanNdArrayStatus terlan_ndarray_v1_quantile_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, double quantile, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation);
TerlanNdArrayStatus terlan_ndarray_v1_percentile_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, double percentile, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation);
TerlanNdArrayStatus terlan_ndarray_v1_quantiles_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, const double *quantiles, int64_t quantile_count, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation);
TerlanNdArrayStatus terlan_ndarray_v1_percentiles_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, const double *percentiles, int64_t percentile_count, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation);
TerlanNdArrayStatus terlan_ndarray_v1_nanquantile_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, double quantile, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation);
TerlanNdArrayStatus terlan_ndarray_v1_nanpercentile_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, double percentile, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation);
TerlanNdArrayStatus terlan_ndarray_v1_nanquantiles_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, const double *quantiles, int64_t quantile_count, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation);
TerlanNdArrayStatus terlan_ndarray_v1_nanpercentiles_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, const double *percentiles, int64_t percentile_count, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int32_t interpolation);
TerlanNdArrayStatus terlan_ndarray_v1_peak_to_peak( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_weighted_average( TerlanNdArrayHandle array, TerlanNdArrayHandle weights, int64_t axis, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_weighted_average_axes(TerlanNdArrayHandle array, TerlanNdArrayHandle weights, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_average_weight_sum(TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_weighted_average_weight_sum(TerlanNdArrayHandle array, TerlanNdArrayHandle weights, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_average_with_sum(TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_weighted_average_with_sum(TerlanNdArrayHandle array, TerlanNdArrayHandle weights, const int64_t *axes, int64_t axis_count, int32_t keep_dims, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_covariance( TerlanNdArrayHandle array, int32_t row_variables, int64_t correction, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_covariance_typed(TerlanNdArrayHandle array, int32_t row_variables, int64_t correction, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_covariance_pair(TerlanNdArrayHandle array, TerlanNdArrayHandle other, int32_t row_variables, int64_t correction, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_covariance_frequency_weighted(TerlanNdArrayHandle array, TerlanNdArrayHandle frequency_weights, int32_t row_variables, int64_t correction, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_covariance_analytic_weighted(TerlanNdArrayHandle array, TerlanNdArrayHandle analytic_weights, int32_t row_variables, int64_t correction, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_covariance_weighted(TerlanNdArrayHandle array, TerlanNdArrayHandle frequency_weights, TerlanNdArrayHandle analytic_weights, int32_t row_variables, int64_t correction, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_covariance_pair_weighted(TerlanNdArrayHandle array, TerlanNdArrayHandle other, TerlanNdArrayHandle frequency_weights, TerlanNdArrayHandle analytic_weights, int32_t row_variables, int64_t correction, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_correlation( TerlanNdArrayHandle array, int32_t row_variables, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_correlation_typed(TerlanNdArrayHandle array, int32_t row_variables, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_correlation_pair(TerlanNdArrayHandle array, TerlanNdArrayHandle other, int32_t row_variables, int32_t dtype, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nan_sum( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nan_product( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nan_mean( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nan_minimum( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nan_maximum( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nan_variance( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int64_t correction, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nan_standard_deviation( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, int64_t correction, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_cumulative_sum( TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_cumulative_product( TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_histogram( TerlanNdArrayHandle array, int64_t bins, double minimum, double maximum, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_histogram_bin_edges( TerlanNdArrayHandle array, int64_t bins, double minimum, double maximum, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_histogramdd( TerlanNdArrayHandle samples, int64_t bins, TerlanNdArrayHandle minima, TerlanNdArrayHandle maxima, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_bincount( TerlanNdArrayHandle values, TerlanNdArrayHandle weights, int64_t minimum_length, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_bincount_unweighted(TerlanNdArrayHandle values, int64_t minimum_length, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_digitize( TerlanNdArrayHandle values, TerlanNdArrayHandle bins, int32_t right, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_difference( TerlanNdArrayHandle array, int64_t order, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_difference_prepend(TerlanNdArrayHandle array, TerlanNdArrayHandle prepend, int64_t order, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_difference_append(TerlanNdArrayHandle array, TerlanNdArrayHandle append, int64_t order, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_difference_surround(TerlanNdArrayHandle array, TerlanNdArrayHandle prepend, TerlanNdArrayHandle append, int64_t order, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_gradient( TerlanNdArrayHandle array, double spacing, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_gradient_edge(TerlanNdArrayHandle array, double spacing, int64_t axis, int32_t edge_order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_gradient_coordinates(TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates, int64_t axis, int32_t edge_order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_gradient_all(TerlanNdArrayHandle array, double spacing, int32_t edge_order, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_gradient_uniform_axes(TerlanNdArrayHandle array, const double *spacings, int64_t spacing_count, const int64_t *axes, int64_t axis_count, int32_t edge_order, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_gradient_axes(TerlanNdArrayHandle array, const uint64_t *spacings, int64_t spacing_count, const int64_t *axes, int64_t axis_count, int32_t edge_order, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_unwrap( TerlanNdArrayHandle array, double period, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_unwrap_discontinuity(TerlanNdArrayHandle array, double period, double discontinuity, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_unwrap_integer_period(TerlanNdArrayHandle array, int64_t period, int64_t discontinuity, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_interpolate( TerlanNdArrayHandle values, TerlanNdArrayHandle coordinates, TerlanNdArrayHandle samples, double left, double right, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_interpolate_default(TerlanNdArrayHandle values, TerlanNdArrayHandle coordinates, TerlanNdArrayHandle samples, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_interpolate_boundaries(TerlanNdArrayHandle values, TerlanNdArrayHandle coordinates, TerlanNdArrayHandle samples, TerlanNdArrayHandle left, TerlanNdArrayHandle right, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_interpolate_periodic(TerlanNdArrayHandle values, TerlanNdArrayHandle coordinates, TerlanNdArrayHandle samples, double period, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_trapezoidal_integral( TerlanNdArrayHandle array, TerlanNdArrayHandle coordinates, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_trapezoidal_integral_uniform( TerlanNdArrayHandle array, double spacing, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_fft( TerlanNdArrayHandle array, int64_t axis, int32_t normalization, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_ifft( TerlanNdArrayHandle array, int64_t axis, int32_t normalization, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_fftn( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t normalization, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_ifftn( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t normalization, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_rfft( TerlanNdArrayHandle array, int64_t axis, int32_t normalization, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_irfft( TerlanNdArrayHandle array, int64_t length, int64_t axis, int32_t normalization, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_rfftn( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t normalization, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_irfftn( TerlanNdArrayHandle array, int64_t length, const int64_t *axes, int64_t axis_count, int32_t normalization, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_hfft( TerlanNdArrayHandle array, int64_t length, int64_t axis, int32_t normalization, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_ihfft( TerlanNdArrayHandle array, int64_t axis, int32_t normalization, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_fftfreq(int64_t length, double spacing, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_rfftfreq(int64_t length, double spacing, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_fftshift( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_ifftshift( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyval( TerlanNdArrayHandle coordinates, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyroots( TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyfit( TerlanNdArrayHandle coordinates, TerlanNdArrayHandle values, int64_t degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyadd( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polysub( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polymul( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polydiv_all( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_polyder( TerlanNdArrayHandle coefficients, int64_t order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyint( TerlanNdArrayHandle coefficients, int64_t order, double constant, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polycompanion( TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyconvert( TerlanNdArrayHandle coefficients, int32_t source_basis, int32_t target_basis, double domain_minimum, double domain_maximum, double window_minimum, double window_maximum, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyfromroots(TerlanNdArrayHandle roots, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyline(double offset, double scale, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polymulx(TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polypow(TerlanNdArrayHandle coefficients, int64_t exponent, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polytrim(TerlanNdArrayHandle coefficients, double tolerance, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyvalfromroots(TerlanNdArrayHandle coordinates, TerlanNdArrayHandle roots, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyvander(TerlanNdArrayHandle coordinates, int64_t degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyval2d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyval3d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle z, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polygrid2d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polygrid3d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle z, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyvander2d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, int64_t x_degree, int64_t y_degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_polyvander3d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle z, int64_t x_degree, int64_t y_degree, int64_t z_degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_tensordot( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, const int64_t *lhs_axes, int64_t lhs_axis_count, const int64_t *rhs_axes, int64_t rhs_axis_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_kron( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_cross( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_einsum( const char *equation, const uint64_t *operands, int64_t operand_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_matmul( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_dot( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_matmul_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs);
TerlanNdArrayStatus terlan_ndarray_v1_dot_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs);
TerlanNdArrayStatus terlan_ndarray_v1_inner( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_outer( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_vdot( TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_trace( TerlanNdArrayHandle array, int64_t offset, int64_t axis_a, int64_t axis_b, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_trace_into(TerlanNdArrayHandle destination, TerlanNdArrayHandle array, int64_t offset, int64_t axis_a, int64_t axis_b);
TerlanNdArrayStatus terlan_ndarray_v1_split( TerlanNdArrayHandle array, int64_t sections, int64_t axis, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_array_split( TerlanNdArrayHandle array, int64_t sections, int64_t axis, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_split_at( TerlanNdArrayHandle array, const int64_t *indices, int64_t index_count, int64_t axis, uint64_t **out_values, size_t *out_length);
void terlan_ndarray_v1_delete_handle_values(uint64_t *values);
TerlanNdArrayStatus terlan_ndarray_v1_count_nonzero( TerlanNdArrayHandle array, int64_t *out_count);
TerlanNdArrayStatus terlan_ndarray_v1_count_nonzero_axes( TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_flatnonzero( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_argwhere( TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nonzero( TerlanNdArrayHandle array, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_export_tensor_packet( TerlanNdArrayHandle array, const char *consumer, uint8_t **out_packet, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_import_tensor_packet( const uint8_t *packet, int64_t packet_length, TerlanNdArrayHandle *out_array);
void terlan_ndarray_v1_delete_tensor_packet(uint8_t *packet);
void terlan_ndarray_v1_delete_bool_values(uint8_t *values);
void terlan_ndarray_v1_delete_int_values(int64_t *values);
void terlan_ndarray_v1_delete_float_values(double *values);
TerlanNdArrayStatus terlan_ndarray_v1_delete(TerlanNdArrayHandle array);
int64_t terlan_ndarray_v1_live_count(void);

TerlanNdArrayStatus terlan_ndarray_v1_cheb2poly(TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_poly2cheb(TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebadd(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebsub(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebmul(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebdiv(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_chebmulx(TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebpow(TerlanNdArrayHandle coefficients, int64_t exponent, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebder(TerlanNdArrayHandle coefficients, int64_t order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebint(TerlanNdArrayHandle coefficients, int64_t order, double constant, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebfit(TerlanNdArrayHandle coordinates, TerlanNdArrayHandle values, int64_t degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebfromroots(TerlanNdArrayHandle roots, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebcompanion(TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebroots(TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebtrim(TerlanNdArrayHandle coefficients, double tolerance, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebline(double offset, double scale, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebval(TerlanNdArrayHandle coordinates, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebval2d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebval3d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle z, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebgrid2d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebgrid3d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle z, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebvander(TerlanNdArrayHandle coordinates, int64_t degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebvander2d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, int64_t x_degree, int64_t y_degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebvander3d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle z, int64_t x_degree, int64_t y_degree, int64_t z_degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebgauss(int64_t degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebweight(TerlanNdArrayHandle coordinates, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebpts1(int64_t count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebpts2(int64_t count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_chebinterpolate(TerlanNdArrayHandle values, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_leg2poly(TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_poly2leg(TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legadd(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legsub(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legmul(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legdiv(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_legmulx(TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legpow(TerlanNdArrayHandle coefficients, int64_t exponent, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legder(TerlanNdArrayHandle coefficients, int64_t order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legint(TerlanNdArrayHandle coefficients, int64_t order, double constant, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legfit(TerlanNdArrayHandle coordinates, TerlanNdArrayHandle values, int64_t degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legfromroots(TerlanNdArrayHandle roots, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legcompanion(TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legroots(TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legtrim(TerlanNdArrayHandle coefficients, double tolerance, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legline(double offset, double scale, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legval(TerlanNdArrayHandle coordinates, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legval2d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legval3d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle z, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_leggrid2d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_leggrid3d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle z, TerlanNdArrayHandle coefficients, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legvander(TerlanNdArrayHandle coordinates, int64_t degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legvander2d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, int64_t x_degree, int64_t y_degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legvander3d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, TerlanNdArrayHandle z, int64_t x_degree, int64_t y_degree, int64_t z_degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_leggauss(int64_t degree, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_legweight(TerlanNdArrayHandle coordinates, TerlanNdArrayHandle *out_array);

TerlanNdArrayStatus terlan_ndarray_v1_allclose(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, double relative_tolerance, double absolute_tolerance, int32_t equal_nan, bool *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_isclose(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, double relative_tolerance, double absolute_tolerance, int32_t equal_nan, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_angle(TerlanNdArrayHandle array, int32_t degrees, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_array_equal(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, bool *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_array_equal_nan(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, bool *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_array_equiv(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, bool *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_asarray_chkfinite(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_asfarray(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_average(TerlanNdArrayHandle array, const int64_t *axes, int64_t axis_count, int32_t keep_dims, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_bartlett(int64_t length, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_blackman(int64_t length, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_hamming(int64_t length, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_hanning(int64_t length, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_kaiser(int64_t length, double beta, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_bitwise_not(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_convolve(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, int64_t mode, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_correlate(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, int64_t mode, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_divmod(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_fix(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_float_power(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_fmod(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_heaviside(TerlanNdArrayHandle lhs, TerlanNdArrayHandle zero_value, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_i0(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_real(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_imag(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_iscomplex(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_isreal(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_iscomplexobj(TerlanNdArrayHandle array, bool *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_isrealobj(TerlanNdArrayHandle array, bool *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_isneginf(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_isposinf(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_ldexp(TerlanNdArrayHandle values, TerlanNdArrayHandle exponents, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nan_to_num(TerlanNdArrayHandle array, double nan, double positive_infinity, double negative_infinity, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_real_if_close(TerlanNdArrayHandle array, double tolerance, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sinc(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_spacing(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_sort_complex(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_trim_zeros(TerlanNdArrayHandle array, int64_t mode, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_unique(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_invert(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_trunc(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);

TerlanNdArrayStatus terlan_ndarray_v1_frombuffer(const uint8_t *bytes, int64_t length, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nanargmax(TerlanNdArrayHandle array, int64_t *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_nanargmin(TerlanNdArrayHandle array, int64_t *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_nancumprod(TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nancumsum(TerlanNdArrayHandle array, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nanmedian(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nanpercentile(TerlanNdArrayHandle array, double percentile, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_nanquantile(TerlanNdArrayHandle array, double quantile, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_packbits(TerlanNdArrayHandle array, int64_t bit_order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_unpackbits(TerlanNdArrayHandle array, int64_t count, int64_t bit_order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_unpackbits_all(TerlanNdArrayHandle array, int64_t bit_order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_packbits_axis(TerlanNdArrayHandle array, int64_t axis, int64_t bit_order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_unpackbits_axis(TerlanNdArrayHandle array, int64_t axis, int64_t count, int64_t bit_order, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_unpackbits_axis_all(TerlanNdArrayHandle array, int64_t axis, int64_t bit_order, TerlanNdArrayHandle *out_array);

TerlanNdArrayStatus terlan_ndarray_v1_alen(TerlanNdArrayHandle array, int64_t *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_append(TerlanNdArrayHandle array, TerlanNdArrayHandle values, int64_t axis, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_block(const uint64_t *arrays, int64_t array_count, const int64_t *layout, int64_t layout_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_broadcast_arrays(TerlanNdArrayHandle first, const uint64_t *remaining, int64_t remaining_count, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_broadcast_shapes(const int64_t *lhs, int64_t lhs_count, const int64_t *rhs, int64_t rhs_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_column_stack(const uint64_t *arrays, int64_t array_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_copyto(TerlanNdArrayHandle destination, TerlanNdArrayHandle source);
TerlanNdArrayStatus terlan_ndarray_v1_copyto_casting(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle source,
    int32_t casting);
TerlanNdArrayStatus terlan_ndarray_v1_copyto_where(
    TerlanNdArrayHandle destination, TerlanNdArrayHandle source,
    TerlanNdArrayHandle mask, int32_t casting);
TerlanNdArrayStatus terlan_ndarray_v1_delete_indexes(TerlanNdArrayHandle array, const int64_t *indexes, int64_t index_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_diag(TerlanNdArrayHandle array, int64_t offset, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_diag_indices(int64_t size, int64_t rank, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_diag_indices_from(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_dsplit(TerlanNdArrayHandle array, int64_t sections, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_dstack(const uint64_t *arrays, int64_t array_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_ediff1d(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_fill_diagonal(TerlanNdArrayHandle array, double value);
TerlanNdArrayStatus terlan_ndarray_v1_full(const int64_t *shape, int64_t rank, double value, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_full_like(TerlanNdArrayHandle array, double value, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogram(TerlanNdArrayHandle array, int64_t bins, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogram_range(TerlanNdArrayHandle array, int64_t bins, double lower, double upper, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogram_edges(TerlanNdArrayHandle array, TerlanNdArrayHandle edges, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogram_weighted(TerlanNdArrayHandle array, TerlanNdArrayHandle weights, int64_t bins, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogram_weighted_range(TerlanNdArrayHandle array, TerlanNdArrayHandle weights, int64_t bins, double lower, double upper, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogram_weighted_edges(TerlanNdArrayHandle array, TerlanNdArrayHandle weights, TerlanNdArrayHandle edges, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogram_estimator(TerlanNdArrayHandle array, int64_t estimator, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogram_estimator_range(TerlanNdArrayHandle array, int64_t estimator, double lower, double upper, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogram_bin_edges(TerlanNdArrayHandle array, int64_t bins, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogram_bin_edges_estimator(TerlanNdArrayHandle array, int64_t estimator, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogramdd(TerlanNdArrayHandle samples, int64_t bins, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogramdd_bins_array(TerlanNdArrayHandle samples, TerlanNdArrayHandle bins, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogramdd_range_arrays(TerlanNdArrayHandle samples, TerlanNdArrayHandle bins, TerlanNdArrayHandle minima, TerlanNdArrayHandle maxima, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogramdd_edges_packed(TerlanNdArrayHandle samples, TerlanNdArrayHandle edges, TerlanNdArrayHandle offsets, TerlanNdArrayHandle dtypes, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogramdd_weighted(TerlanNdArrayHandle samples, TerlanNdArrayHandle weights, int64_t bins, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogramdd_weighted_bins_array(TerlanNdArrayHandle samples, TerlanNdArrayHandle weights, TerlanNdArrayHandle bins, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogramdd_weighted_range_arrays(TerlanNdArrayHandle samples, TerlanNdArrayHandle weights, TerlanNdArrayHandle bins, TerlanNdArrayHandle minima, TerlanNdArrayHandle maxima, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_numpy_histogramdd_weighted_edges_packed(TerlanNdArrayHandle samples, TerlanNdArrayHandle weights, TerlanNdArrayHandle edges, TerlanNdArrayHandle offsets, TerlanNdArrayHandle dtypes, bool density, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_histogram2d(TerlanNdArrayHandle x, TerlanNdArrayHandle y, int64_t bins, TerlanNdArrayHandle minima, TerlanNdArrayHandle maxima, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_hsplit(TerlanNdArrayHandle array, int64_t sections, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_hstack(const uint64_t *arrays, int64_t array_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_insert(TerlanNdArrayHandle array, int64_t index, TerlanNdArrayHandle values, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_ix(TerlanNdArrayHandle first, const uint64_t *remaining, int64_t remaining_count, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_may_share_memory(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, bool *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_meshgrid(TerlanNdArrayHandle first, const uint64_t *remaining, int64_t remaining_count, int64_t indexing, bool sparse, bool copy, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_msort(TerlanNdArrayHandle array, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_pad(TerlanNdArrayHandle array, const int64_t *leading, int64_t leading_count, const int64_t *trailing, int64_t trailing_count, double value, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_place(TerlanNdArrayHandle array, TerlanNdArrayHandle mask, TerlanNdArrayHandle values);
TerlanNdArrayStatus terlan_ndarray_v1_putmask(TerlanNdArrayHandle array, TerlanNdArrayHandle mask, TerlanNdArrayHandle values);
TerlanNdArrayStatus terlan_ndarray_v1_ravel_multi_index(TerlanNdArrayHandle coordinates, const int64_t *dimensions, int64_t rank, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_row_stack(const uint64_t *arrays, int64_t array_count, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_select(const uint64_t *conditions, int64_t condition_count, const uint64_t *choices, int64_t choice_count, TerlanNdArrayHandle default_value, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_shares_memory(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs, bool *out_value);
TerlanNdArrayStatus terlan_ndarray_v1_tril_indices(int64_t rows, int64_t offset, int64_t columns, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_tril_indices_from(TerlanNdArrayHandle array, int64_t offset, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_triu_indices(int64_t rows, int64_t offset, int64_t columns, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_triu_indices_from(TerlanNdArrayHandle array, int64_t offset, TerlanNdArrayHandle *out_array);
TerlanNdArrayStatus terlan_ndarray_v1_unravel_index(TerlanNdArrayHandle indexes, const int64_t *dimensions, int64_t rank, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_vsplit(TerlanNdArrayHandle array, int64_t sections, uint64_t **out_values, size_t *out_length);
TerlanNdArrayStatus terlan_ndarray_v1_vstack(const uint64_t *arrays, int64_t array_count, TerlanNdArrayHandle *out_array);

#ifdef __cplusplus
}
#endif

#endif
