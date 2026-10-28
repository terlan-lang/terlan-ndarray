//! Maintained NumPy-compatible random distributions for owned streams.

use rand::{distr::Open01, Rng};
use rand_distr::{
    Beta, Binomial, Cauchy, ChiSquared, Distribution, Exp, FisherF, Gamma, Geometric, Gumbel,
    Hypergeometric, InverseGaussian, LogNormal, Pareto, Poisson, StandardNormal, StudentT,
    Triangular, Weibull, Zeta,
};

use super::{
    random_checked_integer_output, random_output, random_stream_mut, RandomStream,
    INVALID_ARGUMENT, SIZE_OVERFLOW,
};

type Array = *mut super::super::ffi::TerlanNdArray;
type Stream = *mut super::super::ffi::TerlanNdRandomStream;

unsafe fn float_samples<D: Distribution<f64>>(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    dtype: i32,
    output: *mut Array,
    distribution: Result<D, impl Sized>,
    transform: impl Fn(f64) -> f64,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *output = std::ptr::null_mut() };
    let distribution = match distribution {
        Ok(distribution) => distribution,
        Err(_) => return INVALID_ARGUMENT,
    };
    // SAFETY: generated ownership admits only a live stream handle.
    let stream: &mut RandomStream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    random_output(shape, rank, dtype, output, || {
        transform(distribution.sample(&mut stream.generator))
    })
}

unsafe fn integer_samples<D: Distribution<u64>>(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    dtype: i32,
    output: *mut Array,
    distribution: Result<D, impl Sized>,
    offset: u64,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *output = std::ptr::null_mut() };
    let distribution = match distribution {
        Ok(distribution) => distribution,
        Err(_) => return INVALID_ARGUMENT,
    };
    // SAFETY: generated ownership admits only a live stream handle.
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    random_checked_integer_output(shape, rank, dtype, output, || {
        distribution
            .sample(&mut stream.generator)
            .checked_add(offset)
            .and_then(|value| i64::try_from(value).ok())
            .ok_or(SIZE_OVERFLOW)
    })
}

unsafe fn float_count_samples<D: Distribution<f64>>(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    dtype: i32,
    output: *mut Array,
    distribution: Result<D, impl Sized>,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *output = std::ptr::null_mut() };
    let distribution = match distribution {
        Ok(distribution) => distribution,
        Err(_) => return INVALID_ARGUMENT,
    };
    // SAFETY: generated ownership admits only a live stream handle.
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    random_checked_integer_output(shape, rank, dtype, output, || {
        let value = distribution.sample(&mut stream.generator);
        if value.is_finite() && value >= 0.0 && value <= i64::MAX as f64 {
            Ok(value as i64)
        } else {
            Err(SIZE_OVERFLOW)
        }
    })
}

unsafe fn generated_float_samples(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    dtype: i32,
    output: *mut Array,
    mut sample: impl FnMut(&mut rand_chacha::ChaCha8Rng) -> f64,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *output = std::ptr::null_mut() };
    // SAFETY: generated ownership admits only a live stream handle.
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    random_output(shape, rank, dtype, output, || sample(&mut stream.generator))
}

unsafe fn generated_integer_samples(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    dtype: i32,
    output: *mut Array,
    mut sample: impl FnMut(&mut rand_chacha::ChaCha8Rng) -> Result<i64, i32>,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *output = std::ptr::null_mut() };
    // SAFETY: generated ownership admits only a live stream handle.
    let stream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    random_checked_integer_output(shape, rank, dtype, output, || sample(&mut stream.generator))
}

pub(crate) fn noncentral_chisquare_sample(
    generator: &mut rand_chacha::ChaCha8Rng,
    degrees_of_freedom: f64,
    noncentrality: f64,
) -> f64 {
    let poisson_count = if noncentrality == 0.0 {
        0.0
    } else {
        Poisson::new(noncentrality / 2.0)
            .expect("validated noncentrality creates Poisson")
            .sample(generator)
    };
    ChiSquared::new(degrees_of_freedom + 2.0 * poisson_count)
        .expect("validated degrees create chi-square")
        .sample(generator)
}

pub(crate) fn logarithmic_series_sample(
    generator: &mut rand_chacha::ChaCha8Rng,
    probability: f64,
) -> i64 {
    let log_complement = (-probability).ln_1p();
    loop {
        let candidate: f64 = generator.sample(Open01);
        if candidate >= probability {
            return 1;
        }
        let uniform: f64 = generator.sample(Open01);
        let threshold = -(log_complement * uniform).exp_m1();
        if candidate <= threshold * threshold {
            return (1.0 + candidate.ln() / threshold.ln()).floor() as i64;
        }
        return if candidate >= threshold { 1 } else { 2 };
    }
}

pub(crate) fn von_mises_sample(
    generator: &mut rand_chacha::ChaCha8Rng,
    mean: f64,
    concentration: f64,
) -> f64 {
    use std::f64::consts::{PI, TAU};

    if concentration <= 1.0e-12 {
        let uniform: f64 = generator.sample(Open01);
        return (mean + TAU * (uniform - 0.5) + PI).rem_euclid(TAU) - PI;
    }
    let tau = 1.0 + (1.0 + 4.0 * concentration * concentration).sqrt();
    let rho = (tau - (2.0 * tau).sqrt()) / (2.0 * concentration);
    let ratio = (1.0 + rho * rho) / (2.0 * rho);
    loop {
        let first: f64 = generator.sample(Open01);
        let z = (PI * first).cos();
        let cosine = (1.0 + ratio * z) / (ratio + z);
        let criterion = concentration * (ratio - cosine);
        let second: f64 = generator.sample(Open01);
        if second < criterion * (2.0 - criterion) || second.ln() <= criterion.ln() + 1.0 - criterion
        {
            let third: f64 = generator.sample(Open01);
            let sign = if third >= 0.5 { 1.0 } else { -1.0 };
            return (mean + sign * cosine.clamp(-1.0, 1.0).acos() + PI).rem_euclid(TAU) - PI;
        }
    }
}

macro_rules! float_distribution {
    ($name:ident, $documentation:literal, ($($parameter:ident : $kind:ty),*), $constructor:expr) => {
        #[doc = $documentation]
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            stream: Stream,
            shape: *const i64,
            rank: i64,
            $($parameter: $kind,)*
            dtype: i32,
            output: *mut Array,
        ) -> i32 {
            // SAFETY: all pointers follow the generated borrowed/output ABI contract.
            unsafe { float_samples(stream, shape, rank, dtype, output, $constructor, |value| value) }
        }
    };
}

float_distribution!(terlan_ndarray_v1_random_stream_beta,
    "Samples a beta distribution through `rand_distr`.",
    (alpha: f64, beta: f64), Beta::new(alpha, beta));
float_distribution!(terlan_ndarray_v1_random_stream_chisquare,
    "Samples a chi-square distribution through `rand_distr`.",
    (degrees_of_freedom: f64), ChiSquared::new(degrees_of_freedom));
float_distribution!(terlan_ndarray_v1_random_stream_f,
    "Samples a Fisher F distribution through `rand_distr`.",
    (numerator_degrees: f64, denominator_degrees: f64),
    FisherF::new(numerator_degrees, denominator_degrees));
float_distribution!(terlan_ndarray_v1_random_stream_gamma,
    "Samples a gamma distribution through `rand_distr`.",
    (distribution_shape: f64, scale: f64), Gamma::new(distribution_shape, scale));
float_distribution!(terlan_ndarray_v1_random_stream_gumbel,
    "Samples a Gumbel distribution through `rand_distr`.",
    (location: f64, scale: f64), Gumbel::new(location, scale));
float_distribution!(terlan_ndarray_v1_random_stream_lognormal,
    "Samples a log-normal distribution through `rand_distr`.",
    (mean: f64, sigma: f64), LogNormal::new(mean, sigma));
float_distribution!(terlan_ndarray_v1_random_stream_standard_t,
    "Samples a standard Student t distribution through `rand_distr`.",
    (degrees_of_freedom: f64), StudentT::new(degrees_of_freedom));
float_distribution!(terlan_ndarray_v1_random_stream_triangular,
    "Samples a triangular distribution through `rand_distr`.",
    (left: f64, mode: f64, right: f64), Triangular::new(left, right, mode));
float_distribution!(terlan_ndarray_v1_random_stream_weibull,
    "Samples NumPy's unit-scale Weibull distribution through `rand_distr`.",
    (distribution_shape: f64), Weibull::new(1.0, distribution_shape));

/// Samples NumPy's power distribution through `rand_distr`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_power(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    distribution_shape: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if distribution_shape.is_nan() || distribution_shape == f64::INFINITY {
        let value = if distribution_shape.is_nan() {
            f64::NAN
        } else {
            1.0
        };
        // NumPy consumes one generator value per output even when the parameter
        // makes the transformed result constant or NaN.
        return unsafe {
            generated_float_samples(stream, shape, rank, dtype, output, |generator| {
                let _: f64 = generator.sample(Open01);
                value
            })
        };
    }
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        float_samples(
            stream,
            shape,
            rank,
            dtype,
            output,
            Beta::new(distribution_shape, 1.0),
            |value| value,
        )
    }
}

/// Samples NumPy's zero-based Pareto distribution through `rand_distr`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_pareto(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    distribution_shape: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        float_samples(
            stream,
            shape,
            rank,
            dtype,
            output,
            Pareto::new(1.0, distribution_shape),
            |value| value - 1.0,
        )
    }
}

/// Samples a standard Cauchy distribution through `rand_distr`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_standard_cauchy(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        float_samples(
            stream,
            shape,
            rank,
            dtype,
            output,
            Cauchy::new(0.0, 1.0),
            |v| v,
        )
    }
}

/// Samples an exponential distribution parameterized by NumPy scale.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_exponential(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    scale: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    let distribution = if scale.is_finite() && scale > 0.0 {
        Exp::new(1.0 / scale)
    } else {
        Exp::new(f64::NAN)
    };
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe { float_samples(stream, shape, rank, dtype, output, distribution, |v| v) }
}

/// Samples a unit-scale standard exponential distribution.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_standard_exponential(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe { float_samples(stream, shape, rank, dtype, output, Exp::new(1.0), |v| v) }
}

/// Samples a unit-scale standard gamma distribution.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_standard_gamma(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    distribution_shape: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if distribution_shape == 0.0 {
        if output.is_null() {
            return INVALID_ARGUMENT;
        }
        unsafe { *output = std::ptr::null_mut() };
        if let Err(status) = unsafe { random_stream_mut(stream) } {
            return status;
        }
        return random_output(shape, rank, dtype, output, || 0.0);
    }
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        float_samples(
            stream,
            shape,
            rank,
            dtype,
            output,
            Gamma::new(distribution_shape, 1.0),
            |v| v,
        )
    }
}

/// Samples a standard normal distribution.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_standard_normal(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        float_samples::<StandardNormal>(
            stream,
            shape,
            rank,
            dtype,
            output,
            Ok::<StandardNormal, ()>(StandardNormal),
            |v| v,
        )
    }
}

/// Samples a binomial count distribution through `rand_distr`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_binomial(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    trials: i64,
    probability: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    let distribution = u64::try_from(trials)
        .map_err(|_| ())
        .and_then(|trials| Binomial::new(trials, probability).map_err(|_| ()));
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe { integer_samples(stream, shape, rank, dtype, output, distribution, 0) }
}

/// Samples NumPy geometric trial counts through `rand_distr`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_geometric(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    probability: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    let distribution = if probability > 0.0 && probability <= 1.0 {
        Geometric::new(probability)
    } else {
        Geometric::new(f64::NAN)
    };
    // `rand_distr` returns failures before success; NumPy returns trial count.
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe { integer_samples(stream, shape, rank, dtype, output, distribution, 1) }
}

/// Samples a hypergeometric count distribution through `rand_distr`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_hypergeometric(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    good: i64,
    bad: i64,
    samples: i64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    let distribution = u64::try_from(good)
        .map_err(|_| ())
        .and_then(|good| u64::try_from(bad).map(|bad| (good, bad)).map_err(|_| ()))
        .and_then(|(good, bad)| good.checked_add(bad).map(|total| (total, good)).ok_or(()))
        .and_then(|(total, good)| {
            u64::try_from(samples)
                .map(|samples| (total, good, samples))
                .map_err(|_| ())
        })
        .and_then(|(total, good, samples)| {
            Hypergeometric::new(total, good, samples).map_err(|_| ())
        });
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe { integer_samples(stream, shape, rank, dtype, output, distribution, 0) }
}

/// Samples Poisson counts through `rand_distr`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_poisson(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    lambda: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe { float_count_samples(stream, shape, rank, dtype, output, Poisson::new(lambda)) }
}

/// Samples NumPy Zipf counts through `rand_distr`'s zeta distribution.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_zipf(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    exponent: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe { float_count_samples(stream, shape, rank, dtype, output, Zeta::new(exponent)) }
}

/// Samples a Laplace distribution as the difference of maintained exponentials.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_laplace(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    location: f64,
    scale: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if !location.is_finite() || !scale.is_finite() || scale <= 0.0 {
        if !output.is_null() {
            // SAFETY: the non-null output slot is exclusively owned by this call.
            unsafe { *output = std::ptr::null_mut() };
        }
        return INVALID_ARGUMENT;
    }
    let exponential = Exp::new(1.0 / scale).expect("validated scale creates exponential");
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        generated_float_samples(stream, shape, rank, dtype, output, |generator| {
            location + exponential.sample(generator) - exponential.sample(generator)
        })
    }
}

/// Samples a logistic distribution from open-unit maintained uniform values.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_logistic(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    location: f64,
    scale: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if !location.is_finite() || !scale.is_finite() || scale <= 0.0 {
        if !output.is_null() {
            // SAFETY: the non-null output slot is exclusively owned by this call.
            unsafe { *output = std::ptr::null_mut() };
        }
        return INVALID_ARGUMENT;
    }
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        generated_float_samples(stream, shape, rank, dtype, output, |generator| {
            let probability: f64 = generator.sample(Open01);
            location + scale * (probability / (1.0 - probability)).ln()
        })
    }
}

/// Samples negative-binomial failures through the exact Gamma-Poisson mixture.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_negative_binomial(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    successes: f64,
    probability: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if !successes.is_finite()
        || successes <= 0.0
        || !probability.is_finite()
        || probability <= 0.0
        || probability > 1.0
    {
        if !output.is_null() {
            // SAFETY: the non-null output slot is exclusively owned by this call.
            unsafe { *output = std::ptr::null_mut() };
        }
        return INVALID_ARGUMENT;
    }
    let gamma = if probability == 1.0 {
        None
    } else {
        Some(
            Gamma::new(successes, (1.0 - probability) / probability)
                .expect("validated parameters create gamma"),
        )
    };
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        generated_integer_samples(stream, shape, rank, dtype, output, |generator| {
            let Some(gamma) = gamma.as_ref() else {
                return Ok(0);
            };
            let lambda = gamma.sample(generator);
            let value = Poisson::new(lambda)
                .map_err(|_| INVALID_ARGUMENT)?
                .sample(generator);
            if value <= i64::MAX as f64 {
                Ok(value as i64)
            } else {
                Err(SIZE_OVERFLOW)
            }
        })
    }
}

/// Samples a noncentral chi-square distribution through its Poisson mixture.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_noncentral_chisquare(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    degrees_of_freedom: f64,
    noncentrality: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if !degrees_of_freedom.is_finite()
        || degrees_of_freedom <= 0.0
        || !noncentrality.is_finite()
        || noncentrality < 0.0
    {
        if !output.is_null() {
            // SAFETY: the non-null output slot is exclusively owned by this call.
            unsafe { *output = std::ptr::null_mut() };
        }
        return INVALID_ARGUMENT;
    }
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        generated_float_samples(stream, shape, rank, dtype, output, |generator| {
            noncentral_chisquare_sample(generator, degrees_of_freedom, noncentrality)
        })
    }
}

/// Samples a noncentral Fisher F distribution from maintained chi-square components.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_noncentral_f(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    numerator_degrees: f64,
    denominator_degrees: f64,
    noncentrality: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if !numerator_degrees.is_finite()
        || numerator_degrees <= 0.0
        || !denominator_degrees.is_finite()
        || denominator_degrees <= 0.0
        || !noncentrality.is_finite()
        || noncentrality < 0.0
    {
        if !output.is_null() {
            // SAFETY: the non-null output slot is exclusively owned by this call.
            unsafe { *output = std::ptr::null_mut() };
        }
        return INVALID_ARGUMENT;
    }
    let denominator = ChiSquared::new(denominator_degrees)
        .expect("validated denominator degrees create chi-square");
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        generated_float_samples(stream, shape, rank, dtype, output, |generator| {
            let numerator =
                noncentral_chisquare_sample(generator, numerator_degrees, noncentrality);
            (numerator / numerator_degrees) / (denominator.sample(generator) / denominator_degrees)
        })
    }
}

/// Samples a Rayleigh distribution through the equivalent maintained Weibull form.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_rayleigh(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    scale: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        float_samples(
            stream,
            shape,
            rank,
            dtype,
            output,
            Weibull::new(scale * 2.0_f64.sqrt(), 2.0),
            |value| value,
        )
    }
}

/// Samples a Wald distribution through `rand_distr` inverse Gaussian.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_wald(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    mean: f64,
    scale: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        float_samples(
            stream,
            shape,
            rank,
            dtype,
            output,
            InverseGaussian::new(mean, scale),
            |value| value,
        )
    }
}

/// Samples a logarithmic-series distribution with the Kemp rejection method.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_logseries(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    probability: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if !probability.is_finite() || probability <= 0.0 || probability >= 1.0 {
        if !output.is_null() {
            // SAFETY: the non-null output slot is exclusively owned by this call.
            unsafe { *output = std::ptr::null_mut() };
        }
        return INVALID_ARGUMENT;
    }
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        generated_integer_samples(stream, shape, rank, dtype, output, |generator| {
            Ok(logarithmic_series_sample(generator, probability))
        })
    }
}

/// Samples circular values from a von Mises distribution in `[-pi, pi]`.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_vonmises(
    stream: Stream,
    shape: *const i64,
    rank: i64,
    mean: f64,
    concentration: f64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if !mean.is_finite() || !concentration.is_finite() || concentration < 0.0 {
        if !output.is_null() {
            // SAFETY: the non-null output slot is exclusively owned by this call.
            unsafe { *output = std::ptr::null_mut() };
        }
        return INVALID_ARGUMENT;
    }
    // SAFETY: all pointers follow the generated borrowed/output ABI contract.
    unsafe {
        generated_float_samples(stream, shape, rank, dtype, output, |generator| {
            von_mises_sample(generator, mean, concentration)
        })
    }
}
