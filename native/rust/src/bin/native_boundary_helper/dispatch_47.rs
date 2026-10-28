use super::*;

pub(super) fn accepts_chunk_47(operation: &str) -> bool {
    matches!(operation, "ndarray.numpy.covariance_analytic_weighted" | "ndarray.numpy.covariance_weighted" | "ndarray.numpy.covariance_pair_weighted" | "ndarray.numpy.correlation_typed" | "ndarray.numpy.correlation_pair" | "ndarray.numpy.nansum" | "ndarray.numpy.nanprod" | "ndarray.numpy.nanmean" | "ndarray.numpy.nanmin" | "ndarray.numpy.nanmax" | "ndarray.numpy.nanvar" | "ndarray.numpy.nanstd" | "ndarray.numpy.nansum_as" | "ndarray.numpy.nanprod_as" | "ndarray.numpy.nanmean_as" | "ndarray.numpy.nanvar_as")
}

impl Worker {
    pub(super) fn execute_chunk_47(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.numpy.covariance_analytic_weighted" => {
                let [Arg::Handle(array), Arg::Handle(analytic_weights), Arg::Int(row_variables), Arg::Int(correction), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "covariance_analytic_weighted expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_analytic_weights = match self.live_array(analytic_weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.covariance_analytic_weighted(value_analytic_weights, *row_variables, *correction, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.covariance_weighted" => {
                let [Arg::Handle(array), Arg::Handle(frequency_weights), Arg::Handle(analytic_weights), Arg::Int(row_variables), Arg::Int(correction), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "covariance_weighted expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_frequency_weights = match self.live_array(frequency_weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_analytic_weights = match self.live_array(analytic_weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.covariance_weighted(value_frequency_weights, value_analytic_weights, *row_variables, *correction, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.covariance_pair_weighted" => {
                let [Arg::Handle(array), Arg::Handle(other), Arg::Handle(frequency_weights), Arg::Handle(analytic_weights), Arg::Int(row_variables), Arg::Int(correction), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "covariance_pair_weighted expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_other = match self.live_array(other) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_frequency_weights = match self.live_array(frequency_weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_analytic_weights = match self.live_array(analytic_weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.covariance_pair_weighted(value_other, value_frequency_weights, value_analytic_weights, *row_variables, *correction, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.correlation_typed" => {
                let [Arg::Handle(array), Arg::Int(row_variables), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "correlation_typed expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.correlation_typed(*row_variables, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.correlation_pair" => {
                let [Arg::Handle(array), Arg::Handle(other), Arg::Int(row_variables), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "correlation_pair expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_other = match self.live_array(other) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.correlation_pair(value_other, *row_variables, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nansum" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nansum expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nansum(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanprod" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanprod expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanprod(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanmean" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanmean expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanmean(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanmin" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanmin expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanmin(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanmax" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanmax expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanmax(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanvar" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims), Arg::Int(correction)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanvar expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanvar(arg_ints(axes), *keep_dims, *correction) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanstd" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims), Arg::Int(correction)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanstd expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanstd(arg_ints(axes), *keep_dims, *correction) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nansum_as" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nansum_as expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nansum_as(arg_ints(axes), *keep_dims, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanprod_as" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanprod_as expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanprod_as(arg_ints(axes), *keep_dims, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanmean_as" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanmean_as expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanmean_as(arg_ints(axes), *keep_dims, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanvar_as" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims), Arg::Int(correction), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanvar_as expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanvar_as(arg_ints(axes), *keep_dims, *correction, *dtype) {
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
