use super::*;

pub(super) fn accepts_chunk_40(operation: &str) -> bool {
    matches!(operation, "ndarray.numpy.trim_zeros" | "ndarray.numpy.unique" | "ndarray.numpy.invert" | "ndarray.numpy.trunc" | "ndarray.numpy.frombuffer" | "ndarray.numpy.nanargmax" | "ndarray.numpy.nanargmin" | "ndarray.numpy.nancumprod" | "ndarray.numpy.nancumsum" | "ndarray.numpy.nanmedian" | "ndarray.numpy.nanpercentile" | "ndarray.numpy.nanquantile" | "ndarray.numpy.nanquantile_axes" | "ndarray.numpy.nanpercentile_axes" | "ndarray.numpy.nanquantiles" | "ndarray.numpy.nanpercentiles")
}

impl Worker {
    pub(super) fn execute_chunk_40(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.numpy.trim_zeros" => {
                let [Arg::Handle(array), Arg::Int(mode)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "trim_zeros expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.trim_zeros(*mode) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.unique" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "unique expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.unique() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.invert" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "invert expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.invert() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.trunc" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "trunc expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.trunc() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.frombuffer" => {
                let [Arg::Bytes(bytes)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "frombuffer expects its declared arguments");
                };
                let value = match Array::frombuffer(bytes.as_slice()) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanargmax" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanargmax expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.nanargmax() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nanargmin" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanargmin expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.nanargmin() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nancumprod" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nancumprod expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nancumprod(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nancumsum" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nancumsum expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nancumsum(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanmedian" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanmedian expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanmedian() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanpercentile" => {
                let [Arg::Handle(array), Arg::Float(percentile)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanpercentile expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanpercentile(*percentile) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanquantile" => {
                let [Arg::Handle(array), Arg::Float(quantile)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanquantile expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanquantile(*quantile) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanquantile_axes" => {
                let [Arg::Handle(array), Arg::Float(quantile), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(interpolation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanquantile_axes expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanquantile_axes(*quantile, arg_ints(axes), *keep_dims, *interpolation) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanpercentile_axes" => {
                let [Arg::Handle(array), Arg::Float(percentile), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(interpolation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanpercentile_axes expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanpercentile_axes(*percentile, arg_ints(axes), *keep_dims, *interpolation) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanquantiles" => {
                let [Arg::Handle(array), quantiles @ (Arg::Floats(_) | Arg::EmptyList), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(interpolation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanquantiles expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanquantiles(arg_floats(quantiles), arg_ints(axes), *keep_dims, *interpolation) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanpercentiles" => {
                let [Arg::Handle(array), percentiles @ (Arg::Floats(_) | Arg::EmptyList), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(interpolation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanpercentiles expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanpercentiles(arg_floats(percentiles), arg_ints(axes), *keep_dims, *interpolation) {
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
