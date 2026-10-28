use super::*;

pub(super) fn accepts_chunk_33(operation: &str) -> bool {
    matches!(operation, "ndarray.array.chebmulx" | "ndarray.array.chebpow" | "ndarray.array.chebder" | "ndarray.array.chebint" | "ndarray.array.chebfit" | "ndarray.array.chebfromroots" | "ndarray.array.chebcompanion" | "ndarray.array.chebroots" | "ndarray.array.chebtrim" | "ndarray.array.chebline" | "ndarray.array.chebval" | "ndarray.array.chebval2d" | "ndarray.array.chebval3d" | "ndarray.array.chebgrid2d" | "ndarray.array.chebgrid3d" | "ndarray.array.chebvander")
}

impl Worker {
    pub(super) fn execute_chunk_33(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.chebmulx" => {
                let [Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebmulx expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebmulx(value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebpow" => {
                let [Arg::Handle(coefficients), Arg::Int(exponent)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebpow expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebpow(value_coefficients, *exponent) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebder" => {
                let [Arg::Handle(coefficients), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebder expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebder(value_coefficients, *order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebint" => {
                let [Arg::Handle(coefficients), Arg::Int(order), Arg::Float(constant)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebint expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebint(value_coefficients, *order, *constant) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebfit" => {
                let [Arg::Handle(coordinates), Arg::Handle(values), Arg::Int(degree)] = request.args.as_slice() else {
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
                let value = match chebfit(value_coordinates, value_values, *degree) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebfromroots" => {
                let [Arg::Handle(roots)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebfromroots expects its declared arguments");
                };
                let value_roots = match self.live_array(roots) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebfromroots(value_roots) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebcompanion" => {
                let [Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebcompanion expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebcompanion(value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebroots" => {
                let [Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebroots expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebroots(value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebtrim" => {
                let [Arg::Handle(coefficients), Arg::Float(tolerance)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebtrim expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebtrim(value_coefficients, *tolerance) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebline" => {
                let [Arg::Float(offset), Arg::Float(scale)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebline expects its declared arguments");
                };
                let value = match chebline(*offset, *scale) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebval" => {
                let [Arg::Handle(coordinates), Arg::Handle(coefficients)] = request.args.as_slice() else {
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
                let value = match chebval(value_coordinates, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebval2d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebval2d expects its declared arguments");
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
                let value = match chebval2d(value_x, value_y, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebval3d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(z), Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebval3d expects its declared arguments");
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
                let value = match chebval3d(value_x, value_y, value_z, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebgrid2d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebgrid2d expects its declared arguments");
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
                let value = match chebgrid2d(value_x, value_y, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebgrid3d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(z), Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebgrid3d expects its declared arguments");
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
                let value = match chebgrid3d(value_x, value_y, value_z, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.chebvander" => {
                let [Arg::Handle(coordinates), Arg::Int(degree)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "chebvander expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match chebvander(value_coordinates, *degree) {
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
