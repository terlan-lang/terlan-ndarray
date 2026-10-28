use super::*;

pub(super) fn accepts_chunk_17(operation: &str) -> bool {
    matches!(operation, "ndarray.array.gradient" | "ndarray.array.unwrap" | "ndarray.array.interpolate" | "ndarray.array.trapezoidal_integral" | "ndarray.array.trapezoidal_integral_uniform" | "ndarray.array.fft" | "ndarray.array.fft_sized" | "ndarray.array.ifft" | "ndarray.array.ifft_sized" | "ndarray.array.fftn" | "ndarray.array.ifftn" | "ndarray.array.rfft" | "ndarray.array.rfft_sized" | "ndarray.array.irfft" | "ndarray.array.rfftn" | "ndarray.array.irfftn")
}

impl Worker {
    pub(super) fn execute_chunk_17(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.gradient" => {
                let [Arg::Handle(array), Arg::Float(spacing), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "gradient expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.gradient(*spacing, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.unwrap" => {
                let [Arg::Handle(array), Arg::Float(period), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "unwrap expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.unwrap(*period, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.interpolate" => {
                let [Arg::Handle(values), Arg::Handle(coordinates), Arg::Handle(samples), Arg::Float(left), Arg::Float(right)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "interpolate expects its declared arguments");
                };
                let value_values = match self.live_array(values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_samples = match self.live_array(samples) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_values.interpolate(value_coordinates, value_samples, *left, *right) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.trapezoidal_integral" => {
                let [Arg::Handle(array), Arg::Handle(coordinates), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "trapezoidal_integral expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.trapezoidal_integral(value_coordinates, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.trapezoidal_integral_uniform" => {
                let [Arg::Handle(array), Arg::Float(spacing), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "trapezoidal_integral_uniform expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.trapezoidal_integral_uniform(*spacing, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.fft" => {
                let [Arg::Handle(array), Arg::Int(axis), Arg::Int(normalization)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "fft expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.fft(*axis, *normalization) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.fft_sized" => {
                let [Arg::Handle(array), Arg::Int(length), Arg::Int(axis), Arg::Int(normalization)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "fft expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.fft_sized(*length, *axis, *normalization) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.ifft" => {
                let [Arg::Handle(array), Arg::Int(axis), Arg::Int(normalization)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "ifft expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.ifft(*axis, *normalization) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.ifft_sized" => {
                let [Arg::Handle(array), Arg::Int(length), Arg::Int(axis), Arg::Int(normalization)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "ifft expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.ifft_sized(*length, *axis, *normalization) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.fftn" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(normalization)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "fftn expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.fftn(arg_ints(axes), *normalization) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.ifftn" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(normalization)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "ifftn expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.ifftn(arg_ints(axes), *normalization) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.rfft" => {
                let [Arg::Handle(array), Arg::Int(axis), Arg::Int(normalization)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "rfft expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.rfft(*axis, *normalization) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.rfft_sized" => {
                let [Arg::Handle(array), Arg::Int(length), Arg::Int(axis), Arg::Int(normalization)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "rfft expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.rfft_sized(*length, *axis, *normalization) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.irfft" => {
                let [Arg::Handle(array), Arg::Int(length), Arg::Int(axis), Arg::Int(normalization)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "irfft expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.irfft(*length, *axis, *normalization) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.rfftn" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(normalization)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "rfftn expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.rfftn(arg_ints(axes), *normalization) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.irfftn" => {
                let [Arg::Handle(array), Arg::Int(length), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(normalization)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "irfftn expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.irfftn(*length, arg_ints(axes), *normalization) {
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
