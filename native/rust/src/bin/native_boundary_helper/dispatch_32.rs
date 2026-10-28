use super::*;

pub(super) fn accepts_chunk_32(operation: &str) -> bool {
    matches!(operation, "ndarray.random_stream.choice_uniform_axis" | "ndarray.random_stream.dirichlet" | "ndarray.random_stream.multinomial" | "ndarray.random_stream.multivariate_hypergeometric" | "ndarray.random_stream.multivariate_normal" | "ndarray.random_stream.integers" | "ndarray.random_stream.bernoulli" | "ndarray.random_stream.permutation" | "ndarray.random_stream.shuffle" | "ndarray.random_stream.dispose" | "ndarray.array.cheb2poly" | "ndarray.array.poly2cheb" | "ndarray.array.chebadd" | "ndarray.array.chebsub" | "ndarray.array.chebmul" | "ndarray.array.chebdiv")
}

impl Worker {
    pub(super) fn execute_chunk_32(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.random_stream.choice_uniform_axis" => {
                let [Arg::Handle(stream), Arg::Handle(source), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(replacement), Arg::Int(axis), Arg::Bool(shuffle)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sample_choice_uniform_axis expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                if let Err(error) = self.validate(source, "ndarray.Array.Array") {
                    return error;
                }
                if stream.id == source.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_stream, entry_source] = self.handles.get_disjoint_mut([&stream.id, &source.id]);
                let value_stream = match &mut entry_stream.expect("validated disjoint handle").value {
                    HandleValue::RandomStream(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.RandomStream.RandomStream"),
                };
                let value_source = match &entry_source.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value = match value_stream.sample_choice_uniform_axis(value_source, arg_ints(shape), *replacement, *axis, *shuffle) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.dirichlet" => {
                let [Arg::Handle(stream), Arg::Handle(alpha), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dirichlet expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                if let Err(error) = self.validate(alpha, "ndarray.Array.Array") {
                    return error;
                }
                if stream.id == alpha.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_stream, entry_alpha] = self.handles.get_disjoint_mut([&stream.id, &alpha.id]);
                let value_stream = match &mut entry_stream.expect("validated disjoint handle").value {
                    HandleValue::RandomStream(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.RandomStream.RandomStream"),
                };
                let value_alpha = match &entry_alpha.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value = match value_stream.dirichlet(value_alpha, arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.multinomial" => {
                let [Arg::Handle(stream), Arg::Int(trials), Arg::Handle(probabilities), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "multinomial expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                if let Err(error) = self.validate(probabilities, "ndarray.Array.Array") {
                    return error;
                }
                if stream.id == probabilities.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_stream, entry_probabilities] = self.handles.get_disjoint_mut([&stream.id, &probabilities.id]);
                let value_stream = match &mut entry_stream.expect("validated disjoint handle").value {
                    HandleValue::RandomStream(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.RandomStream.RandomStream"),
                };
                let value_probabilities = match &entry_probabilities.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value = match value_stream.multinomial(*trials, value_probabilities, arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.multivariate_hypergeometric" => {
                let [Arg::Handle(stream), Arg::Handle(colors), Arg::Int(samples), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "multivariate_hypergeometric expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                if let Err(error) = self.validate(colors, "ndarray.Array.Array") {
                    return error;
                }
                if stream.id == colors.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_stream, entry_colors] = self.handles.get_disjoint_mut([&stream.id, &colors.id]);
                let value_stream = match &mut entry_stream.expect("validated disjoint handle").value {
                    HandleValue::RandomStream(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.RandomStream.RandomStream"),
                };
                let value_colors = match &entry_colors.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value = match value_stream.multivariate_hypergeometric(value_colors, *samples, arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.multivariate_normal" => {
                let [Arg::Handle(stream), Arg::Handle(mean), Arg::Handle(covariance), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "multivariate_normal expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                if let Err(error) = self.validate(mean, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(covariance, "ndarray.Array.Array") {
                    return error;
                }
                if stream.id == mean.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if stream.id == covariance.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if mean.id == covariance.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_stream, entry_mean, entry_covariance] = self.handles.get_disjoint_mut([&stream.id, &mean.id, &covariance.id]);
                let value_stream = match &mut entry_stream.expect("validated disjoint handle").value {
                    HandleValue::RandomStream(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.RandomStream.RandomStream"),
                };
                let value_mean = match &entry_mean.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_covariance = match &entry_covariance.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value = match value_stream.multivariate_normal(value_mean, value_covariance, arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.integers" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(low), Arg::Int(high), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "integers expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.integers(arg_ints(shape), *low, *high, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.bernoulli" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(probability)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "bernoulli expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.bernoulli(arg_ints(shape), *probability) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.permutation" => {
                let [Arg::Handle(stream), Arg::Int(length)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "permutation expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.permutation(*length) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.shuffle" => {
                let [Arg::Handle(stream), Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "shuffle expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if stream.id == array.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_stream, entry_array] = self.handles.get_disjoint_mut([&stream.id, &array.id]);
                let value_stream = match &mut entry_stream.expect("validated disjoint handle").value {
                    HandleValue::RandomStream(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.RandomStream.RandomStream"),
                };
                let value_array = match &entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value = match value_stream.shuffle(value_array, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.dispose" => {
                let [Arg::Handle(stream)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dispose expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                self.release_handle(stream.id);
                "ok_unit".to_string()
            }
            "ndarray.array.cheb2poly" => {
                let [Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "cheb2poly expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match cheb2poly(value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.poly2cheb" => {
                let [Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "poly2cheb expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match poly2cheb(value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebadd" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebadd expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebadd(value_lhs, value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebsub" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebsub expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebsub(value_lhs, value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebmul" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebmul expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebmul(value_lhs, value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebdiv" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebdiv expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_lhs.chebdiv(value_rhs) {
                    Ok(values) => values,
                    Err(error) => return native_error(&error),
                };
                let mut handles = Vec::with_capacity(values.len());
                for value in values {
                    let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                        Ok(handle) => handle,
                        Err(error) => return error,
                    };
                    handles.push(format!("{}:{id}:{generation}:{}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array")));
                }
                if handles.is_empty() { "ok_handles".to_string() } else { format!("ok_handles {}", handles.join(",")) }
            }
            _ => unreachable!("operation routed to the wrong generated dispatch chunk"),
        }
    }
}
