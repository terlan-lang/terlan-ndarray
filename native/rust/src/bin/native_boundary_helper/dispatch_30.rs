use super::*;

pub(super) fn accepts_chunk_30(operation: &str) -> bool {
    matches!(operation, "ndarray.random_stream.poisson" | "ndarray.random_stream.standard_normal" | "ndarray.random_stream.standard_normal_into" | "ndarray.random_stream.chisquare" | "ndarray.random_stream.f" | "ndarray.random_stream.gumbel" | "ndarray.random_stream.lognormal" | "ndarray.random_stream.pareto" | "ndarray.random_stream.power" | "ndarray.random_stream.exponential" | "ndarray.random_stream.standard_exponential" | "ndarray.random_stream.standard_exponential_into" | "ndarray.random_stream.standard_gamma" | "ndarray.random_stream.standard_gamma_into" | "ndarray.random_stream.standard_t" | "ndarray.random_stream.triangular")
}

impl Worker {
    pub(super) fn execute_chunk_30(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.random_stream.poisson" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(lambda), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "poisson expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.poisson(arg_ints(shape), *lambda, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.standard_normal" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "standard_normal expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.standard_normal(arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.standard_normal_into" => {
                let [Arg::Handle(stream), Arg::Handle(destination)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "standard_normal_into expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if stream.id == destination.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_stream, entry_destination] = self.handles.get_disjoint_mut([&stream.id, &destination.id]);
                let value_stream = match &mut entry_stream.expect("validated disjoint handle").value {
                    HandleValue::RandomStream(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.RandomStream.RandomStream"),
                };
                let value_destination = match &entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_stream.standard_normal_into(value_destination) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.random_stream.chisquare" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(degrees_of_freedom), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chisquare expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.chisquare(arg_ints(shape), *degrees_of_freedom, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.f" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(numerator_degrees), Arg::Float(denominator_degrees), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "f expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.f(arg_ints(shape), *numerator_degrees, *denominator_degrees, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.gumbel" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(location), Arg::Float(scale), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "gumbel expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.gumbel(arg_ints(shape), *location, *scale, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.lognormal" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(mean), Arg::Float(sigma), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "lognormal expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.lognormal(arg_ints(shape), *mean, *sigma, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.pareto" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(distribution_shape), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pareto expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.pareto(arg_ints(shape), *distribution_shape, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.power" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(distribution_shape), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "power expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.power(arg_ints(shape), *distribution_shape, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.exponential" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(scale), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "exponential expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.exponential(arg_ints(shape), *scale, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.standard_exponential" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "standard_exponential expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.standard_exponential(arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.standard_exponential_into" => {
                let [Arg::Handle(stream), Arg::Handle(destination)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "standard_exponential_into expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if stream.id == destination.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_stream, entry_destination] = self.handles.get_disjoint_mut([&stream.id, &destination.id]);
                let value_stream = match &mut entry_stream.expect("validated disjoint handle").value {
                    HandleValue::RandomStream(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.RandomStream.RandomStream"),
                };
                let value_destination = match &entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_stream.standard_exponential_into(value_destination) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.random_stream.standard_gamma" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(distribution_shape), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "standard_gamma expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.standard_gamma(arg_ints(shape), *distribution_shape, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.standard_gamma_into" => {
                let [Arg::Handle(stream), Arg::Handle(destination), Arg::Handle(distribution_shapes)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "standard_gamma_into expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(distribution_shapes, "ndarray.Array.Array") {
                    return error;
                }
                if stream.id == destination.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if stream.id == distribution_shapes.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if destination.id == distribution_shapes.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_stream, entry_destination, entry_distribution_shapes] = self.handles.get_disjoint_mut([&stream.id, &destination.id, &distribution_shapes.id]);
                let value_stream = match &mut entry_stream.expect("validated disjoint handle").value {
                    HandleValue::RandomStream(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.RandomStream.RandomStream"),
                };
                let value_destination = match &entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_distribution_shapes = match &entry_distribution_shapes.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_stream.standard_gamma_into(value_destination, value_distribution_shapes) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.random_stream.standard_t" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(degrees_of_freedom), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "standard_t expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.standard_t(arg_ints(shape), *degrees_of_freedom, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.triangular" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(left), Arg::Float(mode), Arg::Float(right), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "triangular expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.triangular(arg_ints(shape), *left, *mode, *right, *dtype) {
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
