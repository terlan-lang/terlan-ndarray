use super::*;

pub(super) fn accepts_chunk_29(operation: &str) -> bool {
    matches!(operation, "ndarray.array.kron" | "ndarray.array.cross" | "ndarray.array.einsum" | "ndarray.array.einsum_into" | "ndarray.array.einsum_ordered" | "ndarray.arrow_array.length" | "ndarray.arrow_array.null_count" | "ndarray.arrow_array.dispose" | "ndarray.random_stream.new" | "ndarray.random_stream.sample_broadcast" | "ndarray.random_stream.uniform" | "ndarray.random_stream.uniform_into" | "ndarray.random_stream.normal" | "ndarray.random_stream.beta" | "ndarray.random_stream.gamma" | "ndarray.random_stream.binomial")
}

impl Worker {
    pub(super) fn execute_chunk_29(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.kron" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "kron expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.kron(value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.cross" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "cross expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.cross(value_rhs, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.einsum" => {
                let [operands @ (Arg::Handles(_) | Arg::EmptyList), Arg::String(equation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "einsum expects its declared arguments");
                };
                let value_operands = {
                    let mut values = Vec::new();
                    for handle in arg_handles(operands) {
                        let value = match self.live_array(handle) {
                            Ok(value) => value,
                            Err(error) => return error,
                        };
                        values.push(value);
                    }
                    values
                };
                let value = match einsum(value_operands.as_slice(), equation.as_str()) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.einsum_into" => {
                let [Arg::Handle(destination), operands @ (Arg::Handles(_) | Arg::EmptyList), Arg::String(equation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "einsum_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                for handle in arg_handles(operands) {
                    if let Err(error) = self.validate(handle, "ndarray.Array.Array") {
                        return error;
                    }
                    if handle.id == destination.id {
                        return protocol_error("aliased_mutable_handle", "mutable resource calls cannot borrow the receiver through a resource list");
                    }
                }
                let mut entry_destination = self.handles.remove(&destination.id).expect("validated mutable receiver");
                let value_destination = match &mut entry_destination.value {
                    HandleValue::Array(value) => value,
                    _ => unreachable!("validated mutable receiver type"),
                };
                let value_operands = arg_handles(operands).iter().map(|handle| match &self.handles.get(&handle.id).expect("validated borrowed list resource").value {
                    HandleValue::Array(value) => value,
                    _ => unreachable!("validated borrowed list resource type"),
                }).collect::<Vec<_>>();
                let call_result = value_destination.einsum_into(value_operands.as_slice(), equation.as_str());
                let previous = self.handles.insert(destination.id, entry_destination);
                debug_assert!(previous.is_none());
                match call_result { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.einsum_ordered" => {
                let [operands @ (Arg::Handles(_) | Arg::EmptyList), Arg::String(equation), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "einsum_ordered expects its declared arguments");
                };
                let value_operands = {
                    let mut values = Vec::new();
                    for handle in arg_handles(operands) {
                        let value = match self.live_array(handle) {
                            Ok(value) => value,
                            Err(error) => return error,
                        };
                        values.push(value);
                    }
                    values
                };
                let value = match einsum_ordered(value_operands.as_slice(), equation.as_str(), *order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.arrow_array.length" => {
                let [Arg::Handle(arrow)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "length expects its declared arguments");
                };
                let value_arrow = match self.live_arrowarray(arrow) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_arrow.length() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.arrow_array.null_count" => {
                let [Arg::Handle(arrow)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "null_count expects its declared arguments");
                };
                let value_arrow = match self.live_arrowarray(arrow) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_arrow.null_count() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.arrow_array.dispose" => {
                let [Arg::Handle(arrow)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dispose expects its declared arguments");
                };
                if let Err(error) = self.validate(arrow, "ndarray.ArrowArray.ArrowArray") {
                    return error;
                }
                self.release_handle(arrow.id);
                "ok_unit".to_string()
            }
            "ndarray.random_stream.new" => {
                let [Arg::Int(seed)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "new expects its declared arguments");
                };
                let value = match RandomStream::new(*seed) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::RandomStream(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.RandomStream.RandomStream"))
            }
            "ndarray.random_stream.sample_broadcast" => {
                let [Arg::Handle(stream), parameter_handles @ (Arg::Handles(_) | Arg::EmptyList), Arg::Int(distribution), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sample_broadcast expects its declared arguments");
                };
                if let Err(error) = self.validate(stream, "ndarray.RandomStream.RandomStream") {
                    return error;
                }
                for handle in arg_handles(parameter_handles) {
                    if let Err(error) = self.validate(handle, "ndarray.Array.Array") {
                        return error;
                    }
                    if handle.id == stream.id {
                        return protocol_error("aliased_mutable_handle", "mutable resource calls cannot borrow the receiver through a resource list");
                    }
                }
                let mut entry_stream = self.handles.remove(&stream.id).expect("validated mutable receiver");
                let value_stream = match &mut entry_stream.value {
                    HandleValue::RandomStream(value) => value,
                    _ => unreachable!("validated mutable receiver type"),
                };
                let value_parameter_handles = arg_handles(parameter_handles).iter().map(|handle| match &self.handles.get(&handle.id).expect("validated borrowed list resource").value {
                    HandleValue::Array(value) => value,
                    _ => unreachable!("validated borrowed list resource type"),
                }).collect::<Vec<_>>();
                let call_result = value_stream.sample_broadcast(value_parameter_handles.as_slice(), *distribution, *dtype);
                let previous = self.handles.insert(stream.id, entry_stream);
                debug_assert!(previous.is_none());
                let value = match call_result {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.uniform" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(low), Arg::Float(high), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "uniform expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.uniform(arg_ints(shape), *low, *high, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.uniform_into" => {
                let [Arg::Handle(stream), Arg::Handle(destination)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "uniform_into expects its declared arguments");
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
                match value_stream.uniform_into(value_destination) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.random_stream.normal" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(mean), Arg::Float(standard_deviation), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "normal expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.normal(arg_ints(shape), *mean, *standard_deviation, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.beta" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(alpha), Arg::Float(beta), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "beta expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.beta(arg_ints(shape), *alpha, *beta, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.gamma" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(distribution_shape), Arg::Float(scale), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "gamma expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.gamma(arg_ints(shape), *distribution_shape, *scale, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.random_stream.binomial" => {
                let [Arg::Handle(stream), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(trials), Arg::Float(probability), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "binomial expects its declared arguments");
                };
                let value_stream = match self.live_randomstream_mut(stream) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_stream.binomial(arg_ints(shape), *trials, *probability, *dtype) {
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
