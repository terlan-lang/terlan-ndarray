use super::*;

pub(super) fn accepts_chunk_31(operation: &str) -> bool {
    matches!(operation, "ndarray.random_stream.weibull" | "ndarray.random_stream.geometric" | "ndarray.random_stream.hypergeometric" | "ndarray.random_stream.zipf" | "ndarray.random_stream.standard_cauchy" | "ndarray.random_stream.laplace" | "ndarray.random_stream.logistic" | "ndarray.random_stream.negative_binomial" | "ndarray.random_stream.noncentral_chisquare" | "ndarray.random_stream.noncentral_f" | "ndarray.random_stream.rayleigh" | "ndarray.random_stream.wald" | "ndarray.random_stream.logseries" | "ndarray.random_stream.vonmises" | "ndarray.random_stream.choice" | "ndarray.random_stream.choice_axis")
}

impl Worker {
    pub(super) fn execute_chunk_31(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.random_stream.weibull" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(distribution_shape), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "weibull expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.weibull(arg_ints(shape), *distribution_shape, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.geometric" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(probability), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "geometric expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.geometric(arg_ints(shape), *probability, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.hypergeometric" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(good), Arg::Int(bad), Arg::Int(samples), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "hypergeometric expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.hypergeometric(arg_ints(shape), *good, *bad, *samples, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.zipf" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(exponent), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "zipf expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.zipf(arg_ints(shape), *exponent, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.standard_cauchy" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "standard_cauchy expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.standard_cauchy(arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.laplace" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(location), Arg::Float(scale), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "laplace expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.laplace(arg_ints(shape), *location, *scale, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.logistic" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(location), Arg::Float(scale), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "logistic expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.logistic(arg_ints(shape), *location, *scale, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.negative_binomial" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(successes), Arg::Float(probability), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "negative_binomial expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.negative_binomial(arg_ints(shape), *successes, *probability, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.noncentral_chisquare" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(degrees_of_freedom), Arg::Float(noncentrality), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "noncentral_chisquare expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.noncentral_chisquare(arg_ints(shape), *degrees_of_freedom, *noncentrality, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.noncentral_f" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(numerator_degrees), Arg::Float(denominator_degrees), Arg::Float(noncentrality), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "noncentral_f expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.noncentral_f(arg_ints(shape), *numerator_degrees, *denominator_degrees, *noncentrality, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.rayleigh" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(scale), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "rayleigh expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.rayleigh(arg_ints(shape), *scale, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.wald" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(mean), Arg::Float(scale), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "wald expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.wald(arg_ints(shape), *mean, *scale, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.logseries" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(probability), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "logseries expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.logseries(arg_ints(shape), *probability, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.vonmises" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(mean), Arg::Float(concentration), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "vonmises expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.vonmises(arg_ints(shape), *mean, *concentration, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.choice" => {
                let [Arg::Handle(stream), Arg::Handle(source), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Handle(probabilities), Arg::Bool(replacement)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "choice expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                if let Err(error) = self.validate(source, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(probabilities, "ndarray.Array.Array") {
                    return error;
                }
                if stream.id == source.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if stream.id == probabilities.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if source.id == probabilities.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_stream, entry_source, entry_probabilities] = self.handles.get_disjoint_mut([&stream.id, &source.id, &probabilities.id]);
                let value_stream = match &mut entry_stream.expect("validated disjoint handle").value {
                    HandleValue::RandomStream(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.RandomStream.RandomStream"),
                };
                let value_source = match &entry_source.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_probabilities = match &entry_probabilities.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value = match value_stream.choice(value_source, arg_ints(shape), value_probabilities, *replacement) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.choice_axis" => {
                let [Arg::Handle(stream), Arg::Handle(source), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Handle(probabilities), Arg::Bool(replacement), Arg::Int(axis), Arg::Bool(shuffle)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sample_choice_axis expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                if let Err(error) = self.validate(source, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(probabilities, "ndarray.Array.Array") {
                    return error;
                }
                if stream.id == source.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if stream.id == probabilities.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if source.id == probabilities.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_stream, entry_source, entry_probabilities] = self.handles.get_disjoint_mut([&stream.id, &source.id, &probabilities.id]);
                let value_stream = match &mut entry_stream.expect("validated disjoint handle").value {
                    HandleValue::RandomStream(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.RandomStream.RandomStream"),
                };
                let value_source = match &entry_source.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_probabilities = match &entry_probabilities.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value = match value_stream.sample_choice_axis(value_source, arg_ints(shape), value_probabilities, *replacement, *axis, *shuffle) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            _ => unreachable!("operation routed to the wrong generated dispatch chunk"),
        }
    }
}
