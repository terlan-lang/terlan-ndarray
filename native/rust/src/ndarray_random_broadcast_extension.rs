//! NumPy-compatible broadcasting for array-parameter random distributions.

use rand::{distr::Open01, Rng};
use rand_distr::{
    Beta, Binomial, ChiSquared, Distribution, Exp, FisherF, Gamma, Geometric, Gumbel,
    Hypergeometric, InverseGaussian, LogNormal, Pareto, Poisson, StandardNormal, StudentT,
    Triangular, Weibull, Zeta,
};

use super::native_support::{
    dtype, floats, ints, owned_int_as, owned_real, owned_uint, shape, uints, Array,
    INVALID_ARGUMENT, OK,
};
use super::numpy_shape_extension::handles;
use super::polynomial_power_extension::{broadcast_offset, broadcast_shape, strides};
use super::random_distribution_extension::{
    logarithmic_series_sample, noncentral_chisquare_sample, von_mises_sample,
};
use super::{random_stream_mut, RandomStream, ALLOCATION_FAILED, SIZE_OVERFLOW, UNSUPPORTED_DTYPE};

type Stream = *mut super::super::ffi::TerlanNdRandomStream;

/// Exact signed or unsigned integer values retained from one bound array.
enum ExactIntegerValues {
    /// Values read from Bool or a signed/narrow unsigned dtype.
    Signed(Vec<i64>),
    /// Values read from UInt64 without narrowing through Int64.
    Unsigned(Vec<u64>),
}

impl ExactIntegerValues {
    /// Returns one value in a common lossless comparison representation.
    fn value(&self, index: usize) -> i128 {
        match self {
            Self::Signed(values) => values[index] as i128,
            Self::Unsigned(values) => values[index] as i128,
        }
    }
}

/// One array parameter materialized in logical order with broadcast metadata.
struct BroadcastParameter {
    /// Logical scalar values in row-major iteration order.
    values: Vec<f64>,
    /// Exact logical integer values when required by an integer-bound family.
    integer_values: Option<ExactIntegerValues>,
    /// Logical source shape.
    shape: Vec<i64>,
    /// Row-major strides for the logical source shape.
    strides: Vec<usize>,
}

/// A validated family supported by the generic broadcast sampler.
#[derive(Clone, Copy)]
enum BroadcastDistribution {
    /// Continuous uniform distribution.
    Uniform,
    /// Normal distribution.
    Normal,
    /// Beta distribution.
    Beta,
    /// Gamma distribution.
    Gamma,
    /// NumPy power distribution.
    Power,
    /// Triangular distribution.
    Triangular,
    /// Chi-square distribution.
    ChiSquare,
    /// Fisher F distribution.
    FisherF,
    /// Gumbel distribution.
    Gumbel,
    /// Log-normal distribution.
    LogNormal,
    /// Zero-based Pareto distribution.
    Pareto,
    /// Exponential distribution.
    Exponential,
    /// Unit-scale gamma distribution.
    StandardGamma,
    /// Standard Student t distribution.
    StandardT,
    /// Unit-scale Weibull distribution.
    Weibull,
    /// Laplace distribution.
    Laplace,
    /// Logistic distribution.
    Logistic,
    /// Noncentral chi-square distribution.
    NoncentralChiSquare,
    /// Noncentral Fisher F distribution.
    NoncentralF,
    /// Rayleigh distribution.
    Rayleigh,
    /// Wald inverse-Gaussian distribution.
    Wald,
    /// Circular von Mises distribution.
    VonMises,
    /// Binomial count distribution.
    Binomial,
    /// Geometric trial-count distribution.
    Geometric,
    /// Hypergeometric count distribution.
    Hypergeometric,
    /// Poisson count distribution.
    Poisson,
    /// Zipf count distribution.
    Zipf,
    /// Negative-binomial failure-count distribution.
    NegativeBinomial,
    /// Logarithmic-series count distribution.
    LogSeries,
    /// Half-open bounded integer distribution.
    IntegersOpen,
    /// Closed bounded integer distribution.
    IntegersClosed,
}

impl BroadcastDistribution {
    /// Decodes one stable package-private distribution identifier.
    fn decode(value: i64) -> Result<Self, i32> {
        match value {
            0 => Ok(Self::Uniform),
            1 => Ok(Self::Normal),
            2 => Ok(Self::Beta),
            3 => Ok(Self::Gamma),
            4 => Ok(Self::Power),
            5 => Ok(Self::Triangular),
            6 => Ok(Self::ChiSquare),
            7 => Ok(Self::FisherF),
            8 => Ok(Self::Gumbel),
            9 => Ok(Self::LogNormal),
            10 => Ok(Self::Pareto),
            11 => Ok(Self::Exponential),
            12 => Ok(Self::StandardGamma),
            13 => Ok(Self::StandardT),
            14 => Ok(Self::Weibull),
            15 => Ok(Self::Laplace),
            16 => Ok(Self::Logistic),
            17 => Ok(Self::NoncentralChiSquare),
            18 => Ok(Self::NoncentralF),
            19 => Ok(Self::Rayleigh),
            20 => Ok(Self::Wald),
            21 => Ok(Self::VonMises),
            22 => Ok(Self::Binomial),
            23 => Ok(Self::Geometric),
            24 => Ok(Self::Hypergeometric),
            25 => Ok(Self::Poisson),
            26 => Ok(Self::Zipf),
            27 => Ok(Self::NegativeBinomial),
            28 => Ok(Self::LogSeries),
            29 => Ok(Self::IntegersOpen),
            30 => Ok(Self::IntegersClosed),
            _ => Err(INVALID_ARGUMENT),
        }
    }

    /// Returns the exact number of parameter arrays required by this family.
    fn parameter_count(self) -> usize {
        match self {
            Self::Power
            | Self::ChiSquare
            | Self::Pareto
            | Self::Exponential
            | Self::StandardGamma
            | Self::StandardT
            | Self::Weibull
            | Self::Geometric
            | Self::Poisson
            | Self::Zipf
            | Self::Rayleigh
            | Self::LogSeries => 1,
            Self::Uniform
            | Self::Normal
            | Self::Beta
            | Self::Gamma
            | Self::FisherF
            | Self::Gumbel
            | Self::LogNormal
            | Self::Laplace
            | Self::Logistic
            | Self::NoncentralChiSquare
            | Self::Wald
            | Self::VonMises
            | Self::Binomial
            | Self::NegativeBinomial => 2,
            Self::IntegersOpen | Self::IntegersClosed => 2,
            Self::Triangular | Self::NoncentralF | Self::Hypergeometric => 3,
        }
    }

    /// Returns whether this family produces an integer-valued array.
    fn integer_output(self) -> bool {
        matches!(
            self,
            Self::Binomial
                | Self::Geometric
                | Self::Hypergeometric
                | Self::Poisson
                | Self::Zipf
                | Self::NegativeBinomial
                | Self::LogSeries
                | Self::IntegersOpen
                | Self::IntegersClosed
        )
    }

    /// Returns whether this family requires exact integer parameter storage.
    fn integer_bounds(self) -> bool {
        matches!(self, Self::IntegersOpen | Self::IntegersClosed)
    }

    /// Validates one broadcast parameter tuple before any stream state is consumed.
    fn validates(self, values: &[f64]) -> bool {
        match self {
            Self::Uniform => (values[1] - values[0]).is_finite() && values[0] <= values[1],
            Self::Normal => !values[1].is_nan() && values[1] >= 0.0,
            Self::Beta => Beta::new(values[0], values[1]).is_ok(),
            Self::Gamma => Gamma::new(values[0], values[1]).is_ok(),
            Self::Power => values[0].is_nan() || values[0] > 0.0,
            Self::Triangular => Triangular::new(values[0], values[2], values[1]).is_ok(),
            Self::ChiSquare => ChiSquared::new(values[0]).is_ok(),
            Self::FisherF => FisherF::new(values[0], values[1]).is_ok(),
            Self::Gumbel | Self::Laplace | Self::Logistic => {
                values[0].is_finite() && values[1].is_finite() && values[1] >= 0.0
            }
            Self::LogNormal => values[0].is_finite() && values[1].is_finite() && values[1] >= 0.0,
            Self::Pareto => Pareto::new(1.0, values[0]).is_ok(),
            Self::Exponential | Self::Rayleigh => values[0].is_finite() && values[0] >= 0.0,
            Self::StandardGamma => values[0].is_finite() && values[0] >= 0.0,
            Self::StandardT => StudentT::new(values[0]).is_ok(),
            Self::Weibull => values[0].is_finite() && values[0] >= 0.0,
            Self::NoncentralChiSquare => {
                values[0].is_finite()
                    && values[0] > 0.0
                    && values[1].is_finite()
                    && values[1] >= 0.0
            }
            Self::NoncentralF => {
                values[0].is_finite()
                    && values[0] > 0.0
                    && values[1].is_finite()
                    && values[1] > 0.0
                    && values[2].is_finite()
                    && values[2] >= 0.0
            }
            Self::Wald => InverseGaussian::new(values[0], values[1]).is_ok(),
            Self::VonMises => values[0].is_finite() && values[1].is_finite() && values[1] >= 0.0,
            Self::Binomial => {
                whole_nonnegative(values[0])
                    && values[0] <= u64::MAX as f64
                    && values[1].is_finite()
                    && (0.0..=1.0).contains(&values[1])
            }
            Self::Geometric => values[0].is_finite() && values[0] > 0.0 && values[0] <= 1.0,
            Self::Hypergeometric => hypergeometric_parameters(values).is_some(),
            Self::Poisson => values[0].is_finite() && values[0] >= 0.0,
            Self::Zipf => Zeta::new(values[0]).is_ok(),
            Self::NegativeBinomial => {
                values[0].is_finite()
                    && values[0] > 0.0
                    && values[1].is_finite()
                    && values[1] > 0.0
                    && values[1] <= 1.0
            }
            Self::LogSeries => values[0].is_finite() && values[0] > 0.0 && values[0] < 1.0,
            Self::IntegersOpen | Self::IntegersClosed => {
                unreachable!("integer bounds use exact integer validation")
            }
        }
    }

    /// Samples one already-validated tuple through maintained random primitives.
    fn sample_real(self, values: &[f64], generator: &mut rand_chacha::ChaCha8Rng) -> f64 {
        match self {
            Self::Uniform => values[0] + (values[1] - values[0]) * generator.random::<f64>(),
            Self::Normal => {
                let sample: f64 = StandardNormal.sample(generator);
                values[0] + values[1] * sample
            }
            Self::Beta => Beta::new(values[0], values[1])
                .expect("validated beta parameters")
                .sample(generator),
            Self::Gamma => Gamma::new(values[0], values[1])
                .expect("validated gamma parameters")
                .sample(generator),
            Self::Power if values[0].is_nan() => {
                let _: f64 = generator.sample(Open01);
                f64::NAN
            }
            Self::Power if values[0] == f64::INFINITY => {
                let _: f64 = generator.sample(Open01);
                1.0
            }
            Self::Power => Beta::new(values[0], 1.0)
                .expect("validated power parameter")
                .sample(generator),
            Self::Triangular => Triangular::new(values[0], values[2], values[1])
                .expect("validated triangular parameters")
                .sample(generator),
            Self::ChiSquare => ChiSquared::new(values[0])
                .expect("validated chi-square parameter")
                .sample(generator),
            Self::FisherF => FisherF::new(values[0], values[1])
                .expect("validated Fisher F parameters")
                .sample(generator),
            Self::Gumbel if values[1] == 0.0 => values[0],
            Self::Gumbel => Gumbel::new(values[0], values[1])
                .expect("validated Gumbel parameters")
                .sample(generator),
            Self::LogNormal if values[1] == 0.0 => values[0].exp(),
            Self::LogNormal => LogNormal::new(values[0], values[1])
                .expect("validated log-normal parameters")
                .sample(generator),
            Self::Pareto => {
                Pareto::new(1.0, values[0])
                    .expect("validated Pareto parameter")
                    .sample(generator)
                    - 1.0
            }
            Self::Exponential if values[0] == 0.0 => 0.0,
            Self::Exponential => Exp::new(1.0 / values[0])
                .expect("validated exponential parameter")
                .sample(generator),
            Self::StandardGamma if values[0] == 0.0 => 0.0,
            Self::StandardGamma => Gamma::new(values[0], 1.0)
                .expect("validated standard-gamma parameter")
                .sample(generator),
            Self::StandardT => StudentT::new(values[0])
                .expect("validated Student t parameter")
                .sample(generator),
            Self::Weibull if values[0] == 0.0 => 0.0,
            Self::Weibull => Weibull::new(1.0, values[0])
                .expect("validated Weibull parameter")
                .sample(generator),
            Self::Laplace if values[1] == 0.0 => values[0],
            Self::Laplace => {
                let exponential =
                    Exp::new(1.0 / values[1]).expect("validated Laplace scale creates exponential");
                values[0] + exponential.sample(generator) - exponential.sample(generator)
            }
            Self::Logistic if values[1] == 0.0 => values[0],
            Self::Logistic => {
                let probability: f64 = generator.sample(Open01);
                values[0] + values[1] * (probability / (1.0 - probability)).ln()
            }
            Self::NoncentralChiSquare => {
                noncentral_chisquare_sample(generator, values[0], values[1])
            }
            Self::NoncentralF => {
                let numerator = noncentral_chisquare_sample(generator, values[0], values[2]);
                let denominator = ChiSquared::new(values[1])
                    .expect("validated denominator degrees")
                    .sample(generator);
                (numerator / values[0]) / (denominator / values[1])
            }
            Self::Rayleigh if values[0] == 0.0 => 0.0,
            Self::Rayleigh => Weibull::new(values[0] * 2.0_f64.sqrt(), 2.0)
                .expect("validated Rayleigh scale")
                .sample(generator),
            Self::Wald => InverseGaussian::new(values[0], values[1])
                .expect("validated Wald parameters")
                .sample(generator),
            Self::VonMises => von_mises_sample(generator, values[0], values[1]),
            Self::Binomial
            | Self::Geometric
            | Self::Hypergeometric
            | Self::Poisson
            | Self::Zipf
            | Self::NegativeBinomial
            | Self::LogSeries
            | Self::IntegersOpen
            | Self::IntegersClosed => unreachable!("integer family sampled as real"),
        }
    }

    /// Samples one already-validated integer tuple with checked conversion.
    fn sample_integer(
        self,
        values: &[f64],
        generator: &mut rand_chacha::ChaCha8Rng,
    ) -> Result<i64, i32> {
        match self {
            Self::Binomial => checked_u64(
                Binomial::new(values[0] as u64, values[1])
                    .expect("validated binomial parameters")
                    .sample(generator),
            ),
            Self::Geometric => Geometric::new(values[0])
                .expect("validated geometric parameter")
                .sample(generator)
                .checked_add(1)
                .ok_or(SIZE_OVERFLOW)
                .and_then(checked_u64),
            Self::Hypergeometric => {
                let (total, good, samples) =
                    hypergeometric_parameters(values).expect("validated hypergeometric parameters");
                checked_u64(
                    Hypergeometric::new(total, good, samples)
                        .expect("validated hypergeometric parameters")
                        .sample(generator),
                )
            }
            Self::Poisson if values[0] == 0.0 => Ok(0),
            Self::Poisson => checked_float_count(
                Poisson::new(values[0])
                    .expect("validated Poisson parameter")
                    .sample(generator),
            ),
            Self::Zipf => checked_float_count(
                Zeta::new(values[0])
                    .expect("validated Zipf parameter")
                    .sample(generator),
            ),
            Self::NegativeBinomial if values[1] == 1.0 => Ok(0),
            Self::NegativeBinomial => {
                let lambda = Gamma::new(values[0], (1.0 - values[1]) / values[1])
                    .expect("validated negative-binomial parameters")
                    .sample(generator);
                checked_float_count(
                    Poisson::new(lambda)
                        .map_err(|_| INVALID_ARGUMENT)?
                        .sample(generator),
                )
            }
            Self::LogSeries => Ok(logarithmic_series_sample(generator, values[0])),
            Self::IntegersOpen | Self::IntegersClosed => {
                unreachable!("integer bounds use exact integer sampling")
            }
            _ => unreachable!("real family sampled as integer"),
        }
    }
}

/// Returns whether one floating parameter is exactly representable as a nonnegative integer.
fn whole_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0 && value.fract() == 0.0
}

/// Converts the three hypergeometric parameters to the maintained constructor layout.
fn hypergeometric_parameters(values: &[f64]) -> Option<(u64, u64, u64)> {
    if !values
        .iter()
        .all(|value| whole_nonnegative(*value) && *value <= u64::MAX as f64)
    {
        return None;
    }
    let good = values[0] as u64;
    let bad = values[1] as u64;
    let samples = values[2] as u64;
    good.checked_add(bad).and_then(|total| {
        Hypergeometric::new(total, good, samples)
            .ok()
            .map(|_| (total, good, samples))
    })
}

/// Converts one maintained unsigned count without wrapping.
fn checked_u64(value: u64) -> Result<i64, i32> {
    i64::try_from(value).map_err(|_| SIZE_OVERFLOW)
}

/// Converts one maintained floating count without truncating an invalid result.
fn checked_float_count(value: f64) -> Result<i64, i32> {
    if value.is_finite() && value >= 0.0 && value <= i64::MAX as f64 {
        Ok(value as i64)
    } else {
        Err(SIZE_OVERFLOW)
    }
}

/// Materializes borrowed parameter handles and their broadcast metadata.
unsafe fn parameters(
    values: *const u64,
    count: i64,
    exact_integers: bool,
) -> Result<Vec<BroadcastParameter>, i32> {
    let arrays = unsafe { handles(values, count)? };
    arrays
        .into_iter()
        .map(|array| {
            let shape = unsafe { shape(array)? };
            let strides = strides(&shape)?;
            let (values, integer_values) = if exact_integers {
                let dtype = unsafe { dtype(array)? };
                if !matches!(dtype, 1 | 2 | 4 | 5 | 6 | 7 | 8 | 9 | 10) {
                    return Err(UNSUPPORTED_DTYPE);
                }
                let values = if dtype == 10 {
                    ExactIntegerValues::Unsigned(unsafe { uints(array)? })
                } else {
                    ExactIntegerValues::Signed(unsafe { ints(array)? })
                };
                (Vec::new(), Some(values))
            } else {
                (unsafe { floats(array, false)? }, None)
            };
            Ok(BroadcastParameter {
                values,
                integer_values,
                shape,
                strides,
            })
        })
        .collect()
}

/// Computes one exact integer parameter at a broadcast output offset.
fn parameter_integer_value(
    parameter: &BroadcastParameter,
    flat: usize,
    output_shape: &[i64],
    output_strides: &[usize],
) -> i128 {
    let index = broadcast_offset(
        flat,
        output_shape,
        output_strides,
        &parameter.shape,
        &parameter.strides,
    );
    parameter
        .integer_values
        .as_ref()
        .expect("integer-bound parameters carry exact values")
        .value(index)
}

/// Materializes one exact broadcast integer-bound pair.
fn integer_parameter_tuple(
    parameters: &[BroadcastParameter],
    flat: usize,
    output_shape: &[i64],
    output_strides: &[usize],
) -> [i128; 2] {
    [
        parameter_integer_value(&parameters[0], flat, output_shape, output_strides),
        parameter_integer_value(&parameters[1], flat, output_shape, output_strides),
    ]
}

/// Returns the admitted output range for one integer-compatible dtype code.
fn integer_dtype_bounds(dtype: i32) -> Option<(i128, i128)> {
    match dtype {
        1 => Some((0, 1)),
        2 => Some((i64::MIN as i128, i64::MAX as i128)),
        4 => Some((i8::MIN as i128, i8::MAX as i128)),
        5 => Some((i16::MIN as i128, i16::MAX as i128)),
        6 => Some((i32::MIN as i128, i32::MAX as i128)),
        7 => Some((0, u8::MAX as i128)),
        8 => Some((0, u16::MAX as i128)),
        9 => Some((0, u32::MAX as i128)),
        10 => Some((0, u64::MAX as i128)),
        _ => None,
    }
}

/// Validates one exact bound pair against endpoint and output-dtype semantics.
fn validates_integer_bounds(
    distribution: BroadcastDistribution,
    bounds: [i128; 2],
    dtype: i32,
) -> bool {
    let Some((minimum, maximum)) = integer_dtype_bounds(dtype) else {
        return false;
    };
    let low = bounds[0];
    let high = bounds[1];
    low >= minimum
        && match distribution {
            BroadcastDistribution::IntegersOpen => low < high && high <= maximum + 1,
            BroadcastDistribution::IntegersClosed => low <= high && high <= maximum,
            _ => unreachable!("only integer-bound families reach bound validation"),
        }
}

/// Computes one parameter's logical value at a broadcast output offset.
fn parameter_value(
    parameter: &BroadcastParameter,
    flat: usize,
    output_shape: &[i64],
    output_strides: &[usize],
) -> f64 {
    let index = broadcast_offset(
        flat,
        output_shape,
        output_strides,
        &parameter.shape,
        &parameter.strides,
    );
    parameter.values[index]
}

/// Materializes one broadcast tuple without allocating in the sampling loop.
fn parameter_tuple(
    parameters: &[BroadcastParameter],
    flat: usize,
    output_shape: &[i64],
    output_strides: &[usize],
) -> [f64; 3] {
    let mut values = [0.0; 3];
    for (index, parameter) in parameters.iter().enumerate() {
        values[index] = parameter_value(parameter, flat, output_shape, output_strides);
    }
    values
}

/// Samples one broadcast-compatible set of numerical distribution parameters.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_random_stream_sample_broadcast(
    stream: Stream,
    parameter_handles: *const u64,
    parameter_count: i64,
    distribution: i64,
    dtype: i32,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    // SAFETY: the checked output slot is exclusively owned by this call.
    unsafe { *output = std::ptr::null_mut() };
    let distribution = match BroadcastDistribution::decode(distribution) {
        Ok(distribution) => distribution,
        Err(status) => return status,
    };
    if distribution.integer_output() {
        if !matches!(dtype, 1 | 2 | 4 | 5 | 6 | 7 | 8 | 9 | 10) {
            return UNSUPPORTED_DTYPE;
        }
    } else if !matches!(dtype, 3 | 11 | 14) {
        return UNSUPPORTED_DTYPE;
    }
    let parameters = match unsafe {
        parameters(
            parameter_handles,
            parameter_count,
            distribution.integer_bounds(),
        )
    } {
        Ok(parameters) if parameters.len() == distribution.parameter_count() => parameters,
        Ok(_) => return INVALID_ARGUMENT,
        Err(status) => return status,
    };
    let shapes = parameters
        .iter()
        .map(|parameter| parameter.shape.as_slice())
        .collect::<Vec<_>>();
    let output_shape = match broadcast_shape(&shapes) {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let output_strides = match strides(&output_shape) {
        Ok(strides) => strides,
        Err(status) => return status,
    };
    let count = match super::native_support::element_count(&output_shape) {
        Ok(count) => count,
        Err(status) => return status,
    };
    if distribution.integer_bounds() {
        for flat in 0..count {
            let bounds = integer_parameter_tuple(&parameters, flat, &output_shape, &output_strides);
            if !validates_integer_bounds(distribution, bounds, dtype) {
                return INVALID_ARGUMENT;
            }
        }
    } else {
        for flat in 0..count {
            let tuple = parameter_tuple(&parameters, flat, &output_shape, &output_strides);
            if !distribution.validates(&tuple[..distribution.parameter_count()]) {
                return INVALID_ARGUMENT;
            }
        }
    }
    // Sampling uses a cloned generator so every later failure preserves stream state.
    let stream: &mut RandomStream = match unsafe { random_stream_mut(stream) } {
        Ok(stream) => stream,
        Err(status) => return status,
    };
    let mut generator = stream.generator.clone();
    let array = if distribution.integer_bounds() {
        if dtype == 10 {
            let mut samples = Vec::new();
            if samples.try_reserve_exact(count).is_err() {
                return ALLOCATION_FAILED;
            }
            for flat in 0..count {
                let [low, high] =
                    integer_parameter_tuple(&parameters, flat, &output_shape, &output_strides);
                let sample = match distribution {
                    BroadcastDistribution::IntegersOpen => generator.random_range(low..high),
                    BroadcastDistribution::IntegersClosed => generator.random_range(low..=high),
                    _ => unreachable!("only integer-bound families reach exact sampling"),
                };
                match u64::try_from(sample) {
                    Ok(sample) => samples.push(sample),
                    Err(_) => return INVALID_ARGUMENT,
                }
            }
            match unsafe { owned_uint(&samples, &output_shape) } {
                Ok(array) => array,
                Err(status) => return status,
            }
        } else {
            let mut samples = Vec::new();
            if samples.try_reserve_exact(count).is_err() {
                return ALLOCATION_FAILED;
            }
            for flat in 0..count {
                let [low, high] =
                    integer_parameter_tuple(&parameters, flat, &output_shape, &output_strides);
                let sample = match distribution {
                    BroadcastDistribution::IntegersOpen => generator.random_range(low..high),
                    BroadcastDistribution::IntegersClosed => generator.random_range(low..=high),
                    _ => unreachable!("only integer-bound families reach exact sampling"),
                };
                match i64::try_from(sample) {
                    Ok(sample) => samples.push(sample),
                    Err(_) => return INVALID_ARGUMENT,
                }
            }
            match unsafe { owned_int_as(&samples, &output_shape, dtype) } {
                Ok(array) => array,
                Err(status) => return status,
            }
        }
    } else if distribution.integer_output() {
        let mut samples = Vec::new();
        if samples.try_reserve_exact(count).is_err() {
            return ALLOCATION_FAILED;
        }
        for flat in 0..count {
            let tuple = parameter_tuple(&parameters, flat, &output_shape, &output_strides);
            match distribution
                .sample_integer(&tuple[..distribution.parameter_count()], &mut generator)
            {
                Ok(sample) => samples.push(sample),
                Err(status) => return status,
            }
        }
        match unsafe { owned_int_as(&samples, &output_shape, dtype) } {
            Ok(array) => array,
            Err(status) => return status,
        }
    } else {
        let mut samples = Vec::new();
        if samples.try_reserve_exact(count).is_err() {
            return ALLOCATION_FAILED;
        }
        for flat in 0..count {
            let tuple = parameter_tuple(&parameters, flat, &output_shape, &output_strides);
            samples.push(
                distribution.sample_real(&tuple[..distribution.parameter_count()], &mut generator),
            );
        }
        match unsafe { owned_real(&samples, &output_shape, dtype) } {
            Ok(array) => array,
            Err(status) => return status,
        }
    };
    stream.generator = generator;
    // SAFETY: the owned result is transferred exactly once after stream commit.
    unsafe { *output = array };
    OK
}
