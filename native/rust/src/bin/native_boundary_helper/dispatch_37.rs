use super::*;

pub(super) fn accepts_chunk_37(operation: &str) -> bool {
    matches!(operation, "ndarray.numpy.array_equal_nan" | "ndarray.numpy.array_equiv" | "ndarray.numpy.asarray_chkfinite" | "ndarray.numpy.asarray_chkfinite_as" | "ndarray.numpy.asfarray" | "ndarray.numpy.difference_prepend" | "ndarray.numpy.difference_append" | "ndarray.numpy.difference_surround" | "ndarray.numpy.gradient_edge" | "ndarray.numpy.gradient_coordinates" | "ndarray.numpy.gradient_all" | "ndarray.numpy.gradient_uniform_axes" | "ndarray.numpy.gradient_axes" | "ndarray.numpy.unwrap_discontinuity" | "ndarray.numpy.unwrap_integer_period" | "ndarray.numpy.interpolate_default")
}

impl Worker {
    pub(super) fn execute_chunk_37(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.numpy.array_equal_nan" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "array_equal_nan expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_lhs.array_equal_nan(value_rhs) { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.array_equiv" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "array_equiv expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_lhs.array_equiv(value_rhs) { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.asarray_chkfinite" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "asarray_chkfinite expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.asarray_chkfinite() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.asarray_chkfinite_as" => {
                let [Arg::Handle(array), Arg::Int(requested_dtype), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "asarray_chkfinite expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.asarray_chkfinite_as(*requested_dtype, *order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.asfarray" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "asfarray expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.asfarray() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.difference_prepend" => {
                let [Arg::Handle(array), Arg::Handle(prepend), Arg::Int(order), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "difference_prepend expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_prepend = match self.live_array(prepend) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.difference_prepend(value_prepend, *order, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.difference_append" => {
                let [Arg::Handle(array), Arg::Handle(append), Arg::Int(order), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "difference_append expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_append = match self.live_array(append) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.difference_append(value_append, *order, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.difference_surround" => {
                let [Arg::Handle(array), Arg::Handle(prepend), Arg::Handle(append), Arg::Int(order), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "difference_surround expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_prepend = match self.live_array(prepend) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_append = match self.live_array(append) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.difference_surround(value_prepend, value_append, *order, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.gradient_edge" => {
                let [Arg::Handle(array), Arg::Float(spacing), Arg::Int(axis), Arg::Int(edge_order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "gradient_edge expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.gradient_edge(*spacing, *axis, *edge_order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.gradient_coordinates" => {
                let [Arg::Handle(array), Arg::Handle(coordinates), Arg::Int(axis), Arg::Int(edge_order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "gradient_coordinates expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.gradient_coordinates(value_coordinates, *axis, *edge_order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.gradient_all" => {
                let [Arg::Handle(array), Arg::Float(spacing), Arg::Int(edge_order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "gradient_all expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.gradient_all(*spacing, *edge_order) {
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
            "ndarray.numpy.gradient_uniform_axes" => {
                let [Arg::Handle(array), spacings @ (Arg::Floats(_) | Arg::EmptyList), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(edge_order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "gradient_uniform_axes expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.gradient_uniform_axes(arg_floats(spacings), arg_ints(axes), *edge_order) {
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
            "ndarray.numpy.gradient_axes" => {
                let [Arg::Handle(array), spacings @ (Arg::Handles(_) | Arg::EmptyList), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(edge_order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "gradient_axes expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_spacings = {
                    let mut values = Vec::new();
                    for handle in arg_handles(spacings) {
                        let value = match self.live_array(handle) {
                            Ok(value) => value,
                            Err(error) => return error,
                        };
                        values.push(value);
                    }
                    values
                };
                let values = match value_array.gradient_axes(value_spacings.as_slice(), arg_ints(axes), *edge_order) {
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
            "ndarray.numpy.unwrap_discontinuity" => {
                let [Arg::Handle(array), Arg::Float(period), Arg::Float(discontinuity), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "unwrap_discontinuity expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.unwrap_discontinuity(*period, *discontinuity, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.unwrap_integer_period" => {
                let [Arg::Handle(array), Arg::Int(period), Arg::Int(discontinuity), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "unwrap_integer_period expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.unwrap_integer_period(*period, *discontinuity, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.interpolate_default" => {
                let [Arg::Handle(values), Arg::Handle(coordinates), Arg::Handle(samples)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "interpolate_default expects its declared arguments");
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
                let value = match value_values.interpolate_default(value_coordinates, value_samples) {
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
