use super::*;

pub(super) fn accepts_chunk_36(operation: &str) -> bool {
    matches!(operation, "ndarray.array.leggrid3d" | "ndarray.array.legvander" | "ndarray.array.legvander2d" | "ndarray.array.legvander3d" | "ndarray.array.leggauss" | "ndarray.array.legweight" | "ndarray.array.legder_axis" | "ndarray.array.legint_axis" | "ndarray.array.legval_tensor" | "ndarray.array.legfit_degrees" | "ndarray.array.legfit_weighted_degrees" | "ndarray.numpy.lexsort" | "ndarray.numpy.allclose" | "ndarray.numpy.isclose" | "ndarray.numpy.angle" | "ndarray.numpy.array_equal")
}

impl Worker {
    pub(super) fn execute_chunk_36(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.leggrid3d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(z), Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "leggrid3d expects its declared arguments");
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
                let value = match leggrid3d(value_x, value_y, value_z, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legvander" => {
                let [Arg::Handle(coordinates), Arg::Int(degree)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legvander expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legvander(value_coordinates, *degree) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legvander2d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Int(x_degree), Arg::Int(y_degree)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legvander2d expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legvander2d(value_x, value_y, *x_degree, *y_degree) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legvander3d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(z), Arg::Int(x_degree), Arg::Int(y_degree), Arg::Int(z_degree)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legvander3d expects its declared arguments");
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
                let value = match legvander3d(value_x, value_y, value_z, *x_degree, *y_degree, *z_degree) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.leggauss" => {
                let [Arg::Int(degree)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "leggauss expects its declared arguments");
                };
                let value = match leggauss(*degree) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legweight" => {
                let [Arg::Handle(coordinates)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legweight expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legweight(value_coordinates) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legder_axis" => {
                let [Arg::Handle(coefficients), Arg::Int(order), Arg::Float(scale), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legder expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legder_axis(value_coefficients, *order, *scale, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legint_axis" => {
                let [Arg::Handle(coefficients), Arg::Int(order), Arg::Handle(constants), Arg::Float(lower_bound), Arg::Float(scale), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legint expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_constants = match self.live_array(constants) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legint_axis(value_coefficients, *order, value_constants, *lower_bound, *scale, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legval_tensor" => {
                let [Arg::Handle(coordinates), Arg::Handle(coefficients), Arg::Bool(tensor)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legval expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legval_tensor(value_coordinates, value_coefficients, *tensor) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legfit_degrees" => {
                let [Arg::Handle(coordinates), Arg::Handle(values), Arg::Handle(degrees)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legfit expects its declared arguments");
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
                let value = match legfit_degrees(value_coordinates, value_values, value_degrees) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legfit_weighted_degrees" => {
                let [Arg::Handle(coordinates), Arg::Handle(values), Arg::Handle(degrees), Arg::Handle(weights)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legfit expects its declared arguments");
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
                let value = match legfit_weighted_degrees(value_coordinates, value_values, value_degrees, value_weights) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.lexsort" => {
                let [Arg::Handle(keys), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "lexsort expects its declared arguments");
                };
                let value_keys = match self.live_array(keys) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_keys.lexsort(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.allclose" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs), Arg::Float(relative_tolerance), Arg::Float(absolute_tolerance), Arg::Bool(equal_nan)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "allclose expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_lhs.allclose(value_rhs, *relative_tolerance, *absolute_tolerance, *equal_nan) { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.isclose" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs), Arg::Float(relative_tolerance), Arg::Float(absolute_tolerance), Arg::Bool(equal_nan)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "isclose expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.isclose(value_rhs, *relative_tolerance, *absolute_tolerance, *equal_nan) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.angle" => {
                let [Arg::Handle(array), Arg::Bool(degrees)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "angle expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.angle(*degrees) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.array_equal" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "array_equal expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_lhs.array_equal(value_rhs) { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            _ => unreachable!("operation routed to the wrong generated dispatch chunk"),
        }
    }
}
