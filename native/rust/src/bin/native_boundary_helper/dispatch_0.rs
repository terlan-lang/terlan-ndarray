use super::*;

pub(super) fn accepts_chunk_0(operation: &str) -> bool {
    matches!(operation, "ndarray.array.from_bools" | "ndarray.array.from_ints" | "ndarray.array.from_floats" | "ndarray.array.from_ints_as" | "ndarray.array.from_floats_as" | "ndarray.array.from_complex" | "ndarray.array.empty" | "ndarray.array.zeros" | "ndarray.array.zeros_ordered" | "ndarray.array.subtract_reduce" | "ndarray.array.subtract_accumulate" | "ndarray.array.divide_reduce" | "ndarray.array.divide_accumulate" | "ndarray.array.ones" | "ndarray.array.full_int" | "ndarray.array.full_float")
}

impl Worker {
    pub(super) fn execute_chunk_0(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.from_bools" => {
                let [values @ (Arg::Bools(_) | Arg::EmptyList), shape @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_bools expects its declared arguments");
                };
                let value = match Array::from_bools(arg_bools(values), arg_ints(shape)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.from_ints" => {
                let [values @ (Arg::Ints(_) | Arg::EmptyList), shape @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_ints expects its declared arguments");
                };
                let value = match Array::from_ints(arg_ints(values), arg_ints(shape)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.from_floats" => {
                let [values @ (Arg::Floats(_) | Arg::EmptyList), shape @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_floats expects its declared arguments");
                };
                let value = match Array::from_floats(arg_floats(values), arg_ints(shape)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.from_ints_as" => {
                let [values @ (Arg::Ints(_) | Arg::EmptyList), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_ints_as expects its declared arguments");
                };
                let value = match Array::from_ints_as(arg_ints(values), arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.from_floats_as" => {
                let [values @ (Arg::Floats(_) | Arg::EmptyList), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_floats_as expects its declared arguments");
                };
                let value = match Array::from_floats_as(arg_floats(values), arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.from_complex" => {
                let [real_values @ (Arg::Floats(_) | Arg::EmptyList), imag_values @ (Arg::Floats(_) | Arg::EmptyList), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_complex expects its declared arguments");
                };
                let value = match Array::from_complex(arg_floats(real_values), arg_floats(imag_values), arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.empty" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "empty expects its declared arguments");
                };
                let value = match Array::empty(arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.zeros" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "zeros expects its declared arguments");
                };
                let value = match Array::zeros(arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.zeros_ordered" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "zeros_ordered expects its declared arguments");
                };
                let value = match Array::zeros_ordered(arg_ints(shape), *dtype, *order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.subtract_reduce" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "subtract_reduce expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.subtract_reduce(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.subtract_accumulate" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "subtract_accumulate expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.subtract_accumulate(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.divide_reduce" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "divide_reduce expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.divide_reduce(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.divide_accumulate" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "divide_accumulate expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.divide_accumulate(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.ones" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "ones expects its declared arguments");
                };
                let value = match Array::ones(arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.full_int" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(value), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "full_int expects its declared arguments");
                };
                let value = match Array::full_int(arg_ints(shape), *value, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.full_float" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(value), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "full_float expects its declared arguments");
                };
                let value = match Array::full_float(arg_ints(shape), *value, *dtype) {
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
