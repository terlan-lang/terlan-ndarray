use super::*;

pub(super) fn accepts_chunk_35(operation: &str) -> bool {
    matches!(operation, "ndarray.array.legmul" | "ndarray.array.legdiv" | "ndarray.array.legmulx" | "ndarray.array.legpow" | "ndarray.array.legder" | "ndarray.array.legint" | "ndarray.array.legfit" | "ndarray.array.legfromroots" | "ndarray.array.legcompanion" | "ndarray.array.legroots" | "ndarray.array.legtrim" | "ndarray.array.legline" | "ndarray.array.legval" | "ndarray.array.legval2d" | "ndarray.array.legval3d" | "ndarray.array.leggrid2d")
}

impl Worker {
    pub(super) fn execute_chunk_35(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.legmul" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legmul expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legmul(value_lhs, value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legdiv" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legdiv expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_lhs.legdiv(value_rhs) {
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
            "ndarray.array.legmulx" => {
                let [Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legmulx expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legmulx(value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legpow" => {
                let [Arg::Handle(coefficients), Arg::Int(exponent)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legpow expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legpow(value_coefficients, *exponent) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legder" => {
                let [Arg::Handle(coefficients), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legder expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legder(value_coefficients, *order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legint" => {
                let [Arg::Handle(coefficients), Arg::Int(order), Arg::Float(constant)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legint expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legint(value_coefficients, *order, *constant) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legfit" => {
                let [Arg::Handle(coordinates), Arg::Handle(values), Arg::Int(degree)] = request.args.as_slice() else {
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
                let value = match legfit(value_coordinates, value_values, *degree) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legfromroots" => {
                let [Arg::Handle(roots)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legfromroots expects its declared arguments");
                };
                let value_roots = match self.live_array(roots) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legfromroots(value_roots) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legcompanion" => {
                let [Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legcompanion expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legcompanion(value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legroots" => {
                let [Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legroots expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legroots(value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legtrim" => {
                let [Arg::Handle(coefficients), Arg::Float(tolerance)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legtrim expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match legtrim(value_coefficients, *tolerance) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legline" => {
                let [Arg::Float(offset), Arg::Float(scale)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legline expects its declared arguments");
                };
                let value = match legline(*offset, *scale) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legval" => {
                let [Arg::Handle(coordinates), Arg::Handle(coefficients)] = request.args.as_slice() else {
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
                let value = match legval(value_coordinates, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legval2d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legval2d expects its declared arguments");
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
                let value = match legval2d(value_x, value_y, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.legval3d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(z), Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "legval3d expects its declared arguments");
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
                let value = match legval3d(value_x, value_y, value_z, value_coefficients) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.leggrid2d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(coefficients)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "leggrid2d expects its declared arguments");
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
                let value = match leggrid2d(value_x, value_y, value_coefficients) {
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
