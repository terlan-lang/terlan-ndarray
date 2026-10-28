use super::*;

pub(super) fn accepts_chunk_34(operation: &str) -> bool {
    matches!(operation, "ndarray.array.chebvander2d" | "ndarray.array.chebvander3d" | "ndarray.array.chebgauss" | "ndarray.array.chebweight" | "ndarray.array.chebpts1" | "ndarray.array.chebpts2" | "ndarray.array.chebinterpolate" | "ndarray.array.chebder_axis" | "ndarray.array.chebint_axis" | "ndarray.array.chebval_tensor" | "ndarray.array.chebfit_degrees" | "ndarray.array.chebfit_weighted_degrees" | "ndarray.array.leg2poly" | "ndarray.array.poly2leg" | "ndarray.array.legadd" | "ndarray.array.legsub")
}

impl Worker {
    pub(super) fn execute_chunk_34(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.chebvander2d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Int(x_degree), Arg::Int(y_degree)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebvander2d expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebvander2d(value_x, value_y, *x_degree, *y_degree) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebvander3d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(z), Arg::Int(x_degree), Arg::Int(y_degree), Arg::Int(z_degree)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebvander3d expects its declared arguments");
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
                let value = match chebvander3d(value_x, value_y, value_z, *x_degree, *y_degree, *z_degree) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebgauss" => {
                let [Arg::Int(degree)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebgauss expects its declared arguments");
                };
                let value = match chebgauss(*degree) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebweight" => {
                let [Arg::Handle(coordinates)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebweight expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebweight(value_coordinates) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebpts1" => {
                let [Arg::Int(count)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebpts1 expects its declared arguments");
                };
                let value = match chebpts1(*count) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebpts2" => {
                let [Arg::Int(count)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebpts2 expects its declared arguments");
                };
                let value = match chebpts2(*count) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebinterpolate" => {
                let [Arg::Handle(values)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebinterpolate expects its declared arguments");
                };
                let value_values = match self.live_array(values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebinterpolate(value_values) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebder_axis" => {
                let [Arg::Handle(coefficients), Arg::Int(order), Arg::Float(scale), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebder expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebder_axis(value_coefficients, *order, *scale, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebint_axis" => {
                let [Arg::Handle(coefficients), Arg::Int(order), Arg::Handle(constants), Arg::Float(lower_bound), Arg::Float(scale), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebint expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_constants = match self.live_array(constants) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebint_axis(value_coefficients, *order, value_constants, *lower_bound, *scale, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebval_tensor" => {
                let [Arg::Handle(coordinates), Arg::Handle(coefficients), Arg::Bool(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebval expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebval_tensor(value_coordinates, value_coefficients, *tensor) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebfit_degrees" => {
                let [Arg::Handle(coordinates), Arg::Handle(values), Arg::Handle(degrees)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebfit expects its declared arguments");
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
                let value = match chebfit_degrees(value_coordinates, value_values, value_degrees) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebfit_weighted_degrees" => {
                let [Arg::Handle(coordinates), Arg::Handle(values), Arg::Handle(degrees), Arg::Handle(weights)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebfit expects its declared arguments");
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
                let value = match chebfit_weighted_degrees(value_coordinates, value_values, value_degrees, value_weights) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.leg2poly" => {
                let [Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "leg2poly expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match leg2poly(value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.poly2leg" => {
                let [Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "poly2leg expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match poly2leg(value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legadd" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legadd expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legadd(value_lhs, value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legsub" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legsub expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legsub(value_lhs, value_rhs) {
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
