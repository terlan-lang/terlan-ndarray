use super::*;

pub(super) fn accepts_chunk_38(operation: &str) -> bool {
    matches!(operation, "ndarray.numpy.interpolate_boundaries" | "ndarray.numpy.interpolate_periodic" | "ndarray.numpy.bincount" | "ndarray.numpy.average" | "ndarray.numpy.bartlett" | "ndarray.numpy.blackman" | "ndarray.numpy.hamming" | "ndarray.numpy.hanning" | "ndarray.numpy.kaiser" | "ndarray.numpy.bitwise_not" | "ndarray.numpy.convolve" | "ndarray.numpy.correlate" | "ndarray.numpy.divmod" | "ndarray.numpy.fix" | "ndarray.numpy.float_power" | "ndarray.numpy.fmod")
}

impl Worker {
    pub(super) fn execute_chunk_38(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.numpy.interpolate_boundaries" => {
                let [Arg::Handle(values), Arg::Handle(coordinates), Arg::Handle(samples), Arg::Handle(left), Arg::Handle(right)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "interpolate_boundaries expects its declared arguments");
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
                let value_left = match self.live_array(left) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_right = match self.live_array(right) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_values.interpolate_boundaries(value_coordinates, value_samples, value_left, value_right) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.interpolate_periodic" => {
                let [Arg::Handle(values), Arg::Handle(coordinates), Arg::Handle(samples), Arg::Float(period)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "interpolate_periodic expects its declared arguments");
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
                let value = match value_values.interpolate_periodic(value_coordinates, value_samples, *period) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.bincount" => {
                let [Arg::Handle(values), Arg::Int(minimum_length)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "bincount expects its declared arguments");
                };
                let value_values = match self.live_array(values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_values.numpy_bincount(*minimum_length) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.average" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "average expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.average(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.bartlett" => {
                let [Arg::Int(length)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "bartlett expects its declared arguments");
                };
                let value = match bartlett(*length) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.blackman" => {
                let [Arg::Int(length)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "blackman expects its declared arguments");
                };
                let value = match blackman(*length) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.hamming" => {
                let [Arg::Int(length)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "hamming expects its declared arguments");
                };
                let value = match hamming(*length) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.hanning" => {
                let [Arg::Int(length)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "hanning expects its declared arguments");
                };
                let value = match hanning(*length) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.kaiser" => {
                let [Arg::Int(length), Arg::Float(beta)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "kaiser expects its declared arguments");
                };
                let value = match kaiser(*length, *beta) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.bitwise_not" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "bitwise_not expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.bitwise_not() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.convolve" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs), Arg::Int(mode)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "convolve expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.convolve(value_rhs, *mode) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.correlate" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs), Arg::Int(mode)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "correlate expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.correlate(value_rhs, *mode) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.divmod" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "divmod expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_lhs.divmod(value_rhs) {
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
            "ndarray.numpy.fix" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "fix expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.fix() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.float_power" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "float_power expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.float_power(value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.fmod" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "fmod expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.fmod(value_rhs) {
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
