use super::*;

pub(super) fn accepts_chunk_1(operation: &str) -> bool {
    matches!(operation, "ndarray.array.arange_int_stop" | "ndarray.array.arange_float_stop" | "ndarray.array.arange_int_interval" | "ndarray.array.arange_float_interval" | "ndarray.array.arange_int" | "ndarray.array.arange_float_step" | "ndarray.array.arange" | "ndarray.array.linspace" | "ndarray.array.identity" | "ndarray.array.random_uniform" | "ndarray.array.random_normal" | "ndarray.array.random_integers" | "ndarray.array.random_bernoulli" | "ndarray.array.random_permutation" | "ndarray.array.random_shuffle" | "ndarray.array.diagonal")
}

impl Worker {
    pub(super) fn execute_chunk_1(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.arange_int_stop" => {
                let [Arg::Int(stop)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "arange expects its declared arguments");
                };
                let value = match Array::arange_int_stop(*stop) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.arange_float_stop" => {
                let [Arg::Float(stop)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "arange_float expects its declared arguments");
                };
                let value = match Array::arange_float_stop(*stop) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.arange_int_interval" => {
                let [Arg::Int(start), Arg::Int(stop)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "arange expects its declared arguments");
                };
                let value = match Array::arange_int_interval(*start, *stop) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.arange_float_interval" => {
                let [Arg::Float(start), Arg::Float(stop)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "arange_float expects its declared arguments");
                };
                let value = match Array::arange_float_interval(*start, *stop) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.arange_int" => {
                let [Arg::Int(start), Arg::Int(stop), Arg::Int(step)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "arange_int expects its declared arguments");
                };
                let value = match Array::arange_int(*start, *stop, *step) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.arange_float_step" => {
                let [Arg::Float(start), Arg::Float(stop), Arg::Float(step)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "arange expects its declared arguments");
                };
                let value = match Array::arange_float_step(*start, *stop, *step) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.arange" => {
                let [Arg::Float(start), Arg::Float(stop), Arg::Float(step), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "arange expects its declared arguments");
                };
                let value = match Array::arange_float(*start, *stop, *step, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.linspace" => {
                let [Arg::Float(start), Arg::Float(stop), Arg::Int(count), Arg::Int(endpoint), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "linspace expects its declared arguments");
                };
                let value = match Array::linspace(*start, *stop, *count, *endpoint, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.identity" => {
                let [Arg::Int(rows), Arg::Int(columns), Arg::Int(diagonal), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "identity expects its declared arguments");
                };
                let value = match Array::identity(*rows, *columns, *diagonal, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.random_uniform" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(low), Arg::Float(high), Arg::Int(seed), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "random_uniform expects its declared arguments");
                };
                let value = match Array::random_uniform(arg_ints(shape), *low, *high, *seed, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.random_normal" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(mean), Arg::Float(standard_deviation), Arg::Int(seed), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "random_normal expects its declared arguments");
                };
                let value = match Array::random_normal(arg_ints(shape), *mean, *standard_deviation, *seed, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.random_integers" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(low), Arg::Int(high), Arg::Int(seed), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "random_integers expects its declared arguments");
                };
                let value = match Array::random_integers(arg_ints(shape), *low, *high, *seed, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.random_bernoulli" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(probability), Arg::Int(seed)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "random_bernoulli expects its declared arguments");
                };
                let value = match Array::random_bernoulli(arg_ints(shape), *probability, *seed) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.random_permutation" => {
                let [Arg::Int(length), Arg::Int(seed)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "random_permutation expects its declared arguments");
                };
                let value = match Array::random_permutation(*length, *seed) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.random_shuffle" => {
                let [Arg::Handle(array), Arg::Int(axis), Arg::Int(seed)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "random_shuffle expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.random_shuffle(*axis, *seed) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.diagonal" => {
                let [Arg::Handle(array), Arg::Int(offset), Arg::Int(axis_a), Arg::Int(axis_b)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "diagonal expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.diagonal(*offset, *axis_a, *axis_b) {
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
