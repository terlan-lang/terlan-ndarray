# ndarray.RandomStream

Generated bindings for an owned deterministic ndarray random stream.

## `new`

Creates a deterministic ChaCha8 stream at an explicit seed.

- C symbol: `terlan_ndarray_v1_random_stream_new`
- NativeBoundary operation: `ndarray.random_stream.new`
- Ownership: `opaque_handle`

## `uniform`

Advances the stream while creating uniform floating samples.

- C symbol: `terlan_ndarray_v1_random_stream_uniform`
- NativeBoundary operation: `ndarray.random_stream.uniform`
- Ownership: `opaque_handle`

## `uniform_into`

Advances the stream while atomically filling a C-contiguous floating destination with uniform samples in the half-open interval from zero through one.

- C symbol: `terlan_ndarray_v1_random_stream_uniform_into`
- NativeBoundary operation: `ndarray.random_stream.uniform_into`
- Ownership: `value`

## `normal`

Advances the stream while creating normally distributed floating samples.

- C symbol: `terlan_ndarray_v1_random_stream_normal`
- NativeBoundary operation: `ndarray.random_stream.normal`
- Ownership: `opaque_handle`

## `beta`

Advances the stream while sampling a beta distribution.

- C symbol: `terlan_ndarray_v1_random_stream_beta`
- NativeBoundary operation: `ndarray.random_stream.beta`
- Ownership: `opaque_handle`

## `gamma`

Advances the stream while sampling a gamma distribution.

- C symbol: `terlan_ndarray_v1_random_stream_gamma`
- NativeBoundary operation: `ndarray.random_stream.gamma`
- Ownership: `opaque_handle`

## `binomial`

Advances the stream while sampling binomial counts.

- C symbol: `terlan_ndarray_v1_random_stream_binomial`
- NativeBoundary operation: `ndarray.random_stream.binomial`
- Ownership: `opaque_handle`

## `poisson`

Advances the stream while sampling Poisson counts.

- C symbol: `terlan_ndarray_v1_random_stream_poisson`
- NativeBoundary operation: `ndarray.random_stream.poisson`
- Ownership: `opaque_handle`

## `standard_normal`

Advances the stream while sampling a standard normal distribution.

- C symbol: `terlan_ndarray_v1_random_stream_standard_normal`
- NativeBoundary operation: `ndarray.random_stream.standard_normal`
- Ownership: `opaque_handle`

## `standard_normal_into`

Advances the stream while atomically filling a C-contiguous floating destination with standard normal samples.

- C symbol: `terlan_ndarray_v1_random_stream_standard_normal_into`
- NativeBoundary operation: `ndarray.random_stream.standard_normal_into`
- Ownership: `value`

## `chisquare`

Advances the stream while sampling a chi-square distribution.

- C symbol: `terlan_ndarray_v1_random_stream_chisquare`
- NativeBoundary operation: `ndarray.random_stream.chisquare`
- Ownership: `opaque_handle`

## `f`

Advances the stream while sampling a Fisher F distribution.

- C symbol: `terlan_ndarray_v1_random_stream_f`
- NativeBoundary operation: `ndarray.random_stream.f`
- Ownership: `opaque_handle`

## `gumbel`

Advances the stream while sampling a Gumbel distribution.

- C symbol: `terlan_ndarray_v1_random_stream_gumbel`
- NativeBoundary operation: `ndarray.random_stream.gumbel`
- Ownership: `opaque_handle`

## `lognormal`

Advances the stream while sampling a log-normal distribution.

- C symbol: `terlan_ndarray_v1_random_stream_lognormal`
- NativeBoundary operation: `ndarray.random_stream.lognormal`
- Ownership: `opaque_handle`

## `pareto`

Advances the stream while sampling NumPy's zero-based Pareto distribution.

- C symbol: `terlan_ndarray_v1_random_stream_pareto`
- NativeBoundary operation: `ndarray.random_stream.pareto`
- Ownership: `opaque_handle`

## `power`

Advances the stream while sampling NumPy's power distribution.

- C symbol: `terlan_ndarray_v1_random_stream_power`
- NativeBoundary operation: `ndarray.random_stream.power`
- Ownership: `opaque_handle`

## `exponential`

Advances the stream while sampling an exponential distribution parameterized by scale.

- C symbol: `terlan_ndarray_v1_random_stream_exponential`
- NativeBoundary operation: `ndarray.random_stream.exponential`
- Ownership: `opaque_handle`

## `standard_exponential`

Advances the stream while sampling a unit-scale exponential distribution.

- C symbol: `terlan_ndarray_v1_random_stream_standard_exponential`
- NativeBoundary operation: `ndarray.random_stream.standard_exponential`
- Ownership: `opaque_handle`

## `standard_exponential_into`

Advances the stream while atomically filling a C-contiguous floating destination with unit-scale exponential samples.

- C symbol: `terlan_ndarray_v1_random_stream_standard_exponential_into`
- NativeBoundary operation: `ndarray.random_stream.standard_exponential_into`
- Ownership: `value`

## `standard_gamma`

Advances the stream while sampling a unit-scale gamma distribution.

- C symbol: `terlan_ndarray_v1_random_stream_standard_gamma`
- NativeBoundary operation: `ndarray.random_stream.standard_gamma`
- Ownership: `opaque_handle`

## `standard_gamma_into`

Advances the stream while atomically filling a C-contiguous floating destination with unit-scale gamma samples using broadcast distribution-shape parameters.

- C symbol: `terlan_ndarray_v1_random_stream_standard_gamma_into`
- NativeBoundary operation: `ndarray.random_stream.standard_gamma_into`
- Ownership: `value`

## `standard_t`

Advances the stream while sampling a standard Student t distribution.

- C symbol: `terlan_ndarray_v1_random_stream_standard_t`
- NativeBoundary operation: `ndarray.random_stream.standard_t`
- Ownership: `opaque_handle`

## `triangular`

Advances the stream while sampling a bounded triangular distribution.

- C symbol: `terlan_ndarray_v1_random_stream_triangular`
- NativeBoundary operation: `ndarray.random_stream.triangular`
- Ownership: `opaque_handle`

## `weibull`

Advances the stream while sampling NumPy's unit-scale Weibull distribution.

- C symbol: `terlan_ndarray_v1_random_stream_weibull`
- NativeBoundary operation: `ndarray.random_stream.weibull`
- Ownership: `opaque_handle`

## `geometric`

Advances the stream while sampling one-based geometric trial counts.

- C symbol: `terlan_ndarray_v1_random_stream_geometric`
- NativeBoundary operation: `ndarray.random_stream.geometric`
- Ownership: `opaque_handle`

## `hypergeometric`

Advances the stream while sampling hypergeometric success counts.

- C symbol: `terlan_ndarray_v1_random_stream_hypergeometric`
- NativeBoundary operation: `ndarray.random_stream.hypergeometric`
- Ownership: `opaque_handle`

## `zipf`

Advances the stream while sampling Zipf counts through a zeta distribution.

- C symbol: `terlan_ndarray_v1_random_stream_zipf`
- NativeBoundary operation: `ndarray.random_stream.zipf`
- Ownership: `opaque_handle`

## `standard_cauchy`

Advances the stream while sampling a standard Cauchy distribution.

- C symbol: `terlan_ndarray_v1_random_stream_standard_cauchy`
- NativeBoundary operation: `ndarray.random_stream.standard_cauchy`
- Ownership: `opaque_handle`

## `laplace`

Advances the stream while sampling a Laplace distribution.

- C symbol: `terlan_ndarray_v1_random_stream_laplace`
- NativeBoundary operation: `ndarray.random_stream.laplace`
- Ownership: `opaque_handle`

## `logistic`

Advances the stream while sampling a logistic distribution.

- C symbol: `terlan_ndarray_v1_random_stream_logistic`
- NativeBoundary operation: `ndarray.random_stream.logistic`
- Ownership: `opaque_handle`

## `negative_binomial`

Advances the stream while sampling negative-binomial failure counts.

- C symbol: `terlan_ndarray_v1_random_stream_negative_binomial`
- NativeBoundary operation: `ndarray.random_stream.negative_binomial`
- Ownership: `opaque_handle`

## `noncentral_chisquare`

Advances the stream while sampling a noncentral chi-square distribution.

- C symbol: `terlan_ndarray_v1_random_stream_noncentral_chisquare`
- NativeBoundary operation: `ndarray.random_stream.noncentral_chisquare`
- Ownership: `opaque_handle`

## `noncentral_f`

Advances the stream while sampling a noncentral Fisher F distribution.

- C symbol: `terlan_ndarray_v1_random_stream_noncentral_f`
- NativeBoundary operation: `ndarray.random_stream.noncentral_f`
- Ownership: `opaque_handle`

## `rayleigh`

Advances the stream while sampling a Rayleigh distribution.

- C symbol: `terlan_ndarray_v1_random_stream_rayleigh`
- NativeBoundary operation: `ndarray.random_stream.rayleigh`
- Ownership: `opaque_handle`

## `wald`

Advances the stream while sampling a Wald distribution.

- C symbol: `terlan_ndarray_v1_random_stream_wald`
- NativeBoundary operation: `ndarray.random_stream.wald`
- Ownership: `opaque_handle`

## `logseries`

Advances the stream while sampling positive logarithmic-series counts.

- C symbol: `terlan_ndarray_v1_random_stream_logseries`
- NativeBoundary operation: `ndarray.random_stream.logseries`
- Ownership: `opaque_handle`

## `vonmises`

Advances the stream while sampling circular von Mises values.

- C symbol: `terlan_ndarray_v1_random_stream_vonmises`
- NativeBoundary operation: `ndarray.random_stream.vonmises`
- Ownership: `opaque_handle`

## `choice`

Samples rank-one source values with weighted replacement policy.

- C symbol: `terlan_ndarray_v1_random_stream_choice`
- NativeBoundary operation: `ndarray.random_stream.choice`
- Ownership: `opaque_handle`

## `dirichlet`

Samples simplex vectors parameterized by rank-one positive concentrations.

- C symbol: `terlan_ndarray_v1_random_stream_dirichlet`
- NativeBoundary operation: `ndarray.random_stream.dirichlet`
- Ownership: `opaque_handle`

## `multinomial`

Samples category counts whose final axis sums to the trial count.

- C symbol: `terlan_ndarray_v1_random_stream_multinomial`
- NativeBoundary operation: `ndarray.random_stream.multinomial`
- Ownership: `opaque_handle`

## `multivariate_hypergeometric`

Samples category counts without replacement from a finite population.

- C symbol: `terlan_ndarray_v1_random_stream_multivariate_hypergeometric`
- NativeBoundary operation: `ndarray.random_stream.multivariate_hypergeometric`
- Ownership: `opaque_handle`

## `multivariate_normal`

Samples correlated vectors from a positive-definite covariance matrix.

- C symbol: `terlan_ndarray_v1_random_stream_multivariate_normal`
- NativeBoundary operation: `ndarray.random_stream.multivariate_normal`
- Ownership: `opaque_handle`

## `integers`

Advances the stream while creating bounded integer samples.

- C symbol: `terlan_ndarray_v1_random_stream_integers`
- NativeBoundary operation: `ndarray.random_stream.integers`
- Ownership: `opaque_handle`

## `bernoulli`

Advances the stream while creating Bernoulli samples.

- C symbol: `terlan_ndarray_v1_random_stream_bernoulli`
- NativeBoundary operation: `ndarray.random_stream.bernoulli`
- Ownership: `opaque_handle`

## `permutation`

Advances the stream while creating a permutation of consecutive integers.

- C symbol: `terlan_ndarray_v1_random_stream_permutation`
- NativeBoundary operation: `ndarray.random_stream.permutation`
- Ownership: `opaque_handle`

## `shuffle`

Advances the stream while returning an independently owned shuffled array.

- C symbol: `terlan_ndarray_v1_random_stream_shuffle`
- NativeBoundary operation: `ndarray.random_stream.shuffle`
- Ownership: `opaque_handle`

## `dispose`

Disposes an owned stream exactly once.

- C symbol: `terlan_ndarray_v1_random_stream_delete`
- NativeBoundary operation: `ndarray.random_stream.dispose`
- Ownership: `dispose_handle`
