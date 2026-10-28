use super::*;

pub(super) fn accepts_chunk_26(operation: &str) -> bool {
    matches!(operation, "ndarray.array.symmetric_eigen_general" | "ndarray.array.cond" | "ndarray.array.eig" | "ndarray.array.condition" | "ndarray.array.eigvals" | "ndarray.array.eigvalsh" | "ndarray.array.eigvalsh_general" | "ndarray.array.matrix_power" | "ndarray.array.multi_dot" | "ndarray.array.multi_dot_into" | "ndarray.array.norm" | "ndarray.array.norm_general" | "ndarray.array.pinv" | "ndarray.array.slogdet" | "ndarray.array.tensorinv" | "ndarray.array.tensorsolve")
}

impl Worker {
    pub(super) fn execute_chunk_26(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.symmetric_eigen_general" => {
                let [Arg::Handle(array), Arg::Float(tolerance), Arg::Bool(upper)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "symmetric_eigen_general expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.symmetric_eigen_general(*tolerance, *upper) {
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
            "ndarray.array.cond" => {
                let [Arg::Handle(array), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "cond expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.cond(*order) { Ok(value) => format!("ok_float {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.eig" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "eig expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.eig() {
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
            "ndarray.array.condition" => {
                let [Arg::Handle(array), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "condition expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.condition(*order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.eigvals" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "eigvals expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.eigvals() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.eigvalsh" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "eigvalsh expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.eigvalsh() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.eigvalsh_general" => {
                let [Arg::Handle(array), Arg::Bool(upper)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "eigvalsh_general expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.eigvalsh_general(*upper) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.matrix_power" => {
                let [Arg::Handle(array), Arg::Int(exponent)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "matrix_power expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.matrix_power(*exponent) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.multi_dot" => {
                let [arrays @ (Arg::Handles(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "multi_dot expects its declared arguments");
                };
                let value_arrays = {
                    let mut values = Vec::new();
                    for handle in arg_handles(arrays) {
                        let value = match self.live_array(handle) {
                            Ok(value) => value,
                            Err(error) => return error,
                        };
                        values.push(value);
                    }
                    values
                };
                let value = match multi_dot(value_arrays.as_slice()) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.multi_dot_into" => {
                let [Arg::Handle(destination), arrays @ (Arg::Handles(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "multi_dot_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                for handle in arg_handles(arrays) {
                    if let Err(error) = self.validate(handle, "ndarray.Array.Array") {
                        return error;
                    }
                    if handle.id == destination.id {
                        return protocol_error("aliased_mutable_handle", "mutable resource calls cannot borrow the receiver through a resource list");
                    }
                }
                let mut entry_destination = self.handles.remove(&destination.id).expect("validated mutable receiver");
                let value_destination = match &mut entry_destination.value {
                    HandleValue::Array(value) => value,
                    _ => unreachable!("validated mutable receiver type"),
                };
                let value_arrays = arg_handles(arrays).iter().map(|handle| match &self.handles.get(&handle.id).expect("validated borrowed list resource").value {
                    HandleValue::Array(value) => value,
                    _ => unreachable!("validated borrowed list resource type"),
                }).collect::<Vec<_>>();
                let call_result = value_destination.multi_dot_into(value_arrays.as_slice());
                let previous = self.handles.insert(destination.id, entry_destination);
                debug_assert!(previous.is_none());
                match call_result { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.norm" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "norm expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.norm() { Ok(value) => format!("ok_float {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.norm_general" => {
                let [Arg::Handle(array), Arg::Int(order_kind), Arg::Float(order_value), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keepdims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "norm_general expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.norm_general(*order_kind, *order_value, arg_ints(axes), *keepdims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.pinv" => {
                let [Arg::Handle(array), Arg::Float(tolerance)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pinv expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.pinv(*tolerance) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.slogdet" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "slogdet expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.slogdet() {
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
            "ndarray.array.tensorinv" => {
                let [Arg::Handle(array), Arg::Int(index)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "tensorinv expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.tensorinv(*index) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.tensorsolve" => {
                let [Arg::Handle(coefficients), Arg::Handle(rhs), axes @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "tensorsolve expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_coefficients.tensorsolve(value_rhs, arg_ints(axes)) {
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
