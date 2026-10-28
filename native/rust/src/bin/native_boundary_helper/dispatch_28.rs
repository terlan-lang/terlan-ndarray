use super::*;

pub(super) fn accepts_chunk_28(operation: &str) -> bool {
    matches!(operation, "ndarray.array.polyvalfromroots" | "ndarray.array.polyder_axis" | "ndarray.array.polyint_axis" | "ndarray.array.polyvalfromroots_tensor" | "ndarray.array.polyfit_degrees" | "ndarray.array.polyfit_weighted_degrees" | "ndarray.array.polyfit_covariance" | "ndarray.array.polyfit_weighted_covariance" | "ndarray.array.polyvander" | "ndarray.array.polyval2d" | "ndarray.array.polyval3d" | "ndarray.array.polygrid2d" | "ndarray.array.polygrid3d" | "ndarray.array.polyvander2d" | "ndarray.array.polyvander3d" | "ndarray.array.tensordot")
}

impl Worker {
    pub(super) fn execute_chunk_28(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.polyvalfromroots" => {
                let [Arg::Handle(coordinates), Arg::Handle(roots)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyvalfromroots expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_roots = match self.live_array(roots) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_coordinates.polyvalfromroots(value_roots) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.polyder_axis" => {
                let [Arg::Handle(coefficients), Arg::Int(order), Arg::Float(scale), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyder_axis expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_coefficients.polyder_axis(*order, *scale, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.polyint_axis" => {
                let [Arg::Handle(coefficients), Arg::Int(order), Arg::Handle(constants), Arg::Float(lower_bound), Arg::Float(scale), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyint_axis expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_constants = match self.live_array(constants) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_coefficients.polyint_axis(*order, value_constants, *lower_bound, *scale, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.polyvalfromroots_tensor" => {
                let [Arg::Handle(coordinates), Arg::Handle(roots), Arg::Bool(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyvalfromroots_tensor expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_roots = match self.live_array(roots) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_coordinates.polyvalfromroots_tensor(value_roots, *tensor) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.polyfit_degrees" => {
                let [Arg::Handle(coordinates), Arg::Handle(values), Arg::Handle(degrees)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyfit_degrees expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_values = match self.live_array(values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_degrees = match self.live_array(degrees) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_coordinates.polyfit_degrees(value_values, value_degrees) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.polyfit_weighted_degrees" => {
                let [Arg::Handle(coordinates), Arg::Handle(values), Arg::Handle(degrees), Arg::Handle(weights)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyfit_weighted_degrees expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_values = match self.live_array(values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_degrees = match self.live_array(degrees) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weights = match self.live_array(weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_coordinates.polyfit_weighted_degrees(value_values, value_degrees, value_weights) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.polyfit_covariance" => {
                let [Arg::Handle(coordinates), Arg::Handle(values), Arg::Int(degree), Arg::Bool(scale_covariance)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyfit_covariance expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_values = match self.live_array(values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_coordinates.polyfit_covariance(value_values, *degree, *scale_covariance) {
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
            "ndarray.array.polyfit_weighted_covariance" => {
                let [Arg::Handle(coordinates), Arg::Handle(values), Arg::Int(degree), Arg::Handle(weights), Arg::Bool(scale_covariance)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyfit_weighted_covariance expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_values = match self.live_array(values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weights = match self.live_array(weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_coordinates.polyfit_weighted_covariance(value_values, *degree, value_weights, *scale_covariance) {
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
            "ndarray.array.polyvander" => {
                let [Arg::Handle(coordinates), Arg::Int(degree)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyvander expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_coordinates.polyvander(*degree) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.polyval2d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyval2d expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match polyval2d(value_x, value_y, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.polyval3d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(z), Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyval3d expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_z = match self.live_array(z) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match polyval3d(value_x, value_y, value_z, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.polygrid2d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polygrid2d expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match polygrid2d(value_x, value_y, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.polygrid3d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(z), Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polygrid3d expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_z = match self.live_array(z) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match polygrid3d(value_x, value_y, value_z, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.polyvander2d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Int(x_degree), Arg::Int(y_degree)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyvander2d expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match polyvander2d(value_x, value_y, *x_degree, *y_degree) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.polyvander3d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(z), Arg::Int(x_degree), Arg::Int(y_degree), Arg::Int(z_degree)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "polyvander3d expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_z = match self.live_array(z) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match polyvander3d(value_x, value_y, value_z, *x_degree, *y_degree, *z_degree) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.tensordot" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs), lhs_axes @ (Arg::Ints(_) | Arg::EmptyList), rhs_axes @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "tensordot expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.tensordot(value_rhs, arg_ints(lhs_axes), arg_ints(rhs_axes)) {
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
