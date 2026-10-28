use super::*;

pub(super) fn accepts_chunk_14(operation: &str) -> bool {
    matches!(operation, "ndarray.array.radians" | "ndarray.array.sign_bit" | "ndarray.array.frexp_all" | "ndarray.array.modf_all" | "ndarray.array.sum" | "ndarray.array.product" | "ndarray.array.mean" | "ndarray.array.mean_where" | "ndarray.array.mean_as" | "ndarray.array.mean_into" | "ndarray.array.minimum" | "ndarray.array.maximum" | "ndarray.array.any" | "ndarray.array.all" | "ndarray.array.argmin" | "ndarray.array.argmax")
}

impl Worker {
    pub(super) fn execute_chunk_14(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.radians" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "radians expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.radians() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.sign_bit" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sign_bit expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.sign_bit() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.frexp_all" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "frexp_all expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.frexp_all() {
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
            "ndarray.array.modf_all" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "modf_all expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.modf_all() {
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
            "ndarray.array.sum" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sum expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.sum(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.product" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "product expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.product(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.mean" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "mean expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.mean(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.mean_where" => {
                let [Arg::Handle(array), Arg::Handle(mask), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "mean_where expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_mask = match self.live_array(mask) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.mean_where(value_mask, arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.mean_as" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "mean_as expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.mean_as(arg_ints(axes), *keep_dims, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.mean_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "mean_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if destination.id == array.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_destination, entry_array] = self.handles.get_disjoint_mut([&destination.id, &array.id]);
                let value_destination = match &mut entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_array = match &entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_destination.mean_into(value_array, arg_ints(axes), *keep_dims) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.minimum" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "minimum expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.minimum(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.maximum" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "maximum expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.maximum(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.any" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "any expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.any(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.all" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "all expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.all(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.argmin" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "argmin expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.argmin(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.argmax" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "argmax expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.argmax(arg_ints(axes), *keep_dims) {
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
