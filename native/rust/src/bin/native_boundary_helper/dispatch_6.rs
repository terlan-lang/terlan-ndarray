use super::*;

pub(super) fn accepts_chunk_6(operation: &str) -> bool {
    matches!(operation, "ndarray.array.matrix_rank" | "ndarray.array.to_npy" | "ndarray.array.from_npy" | "ndarray.array.save_npy" | "ndarray.array.load_npy" | "ndarray.array.serialize" | "ndarray.array.deserialize" | "ndarray.array.mmap_npy" | "ndarray.array.flush" | "ndarray.array.to_arrow" | "ndarray.array.from_arrow" | "ndarray.array.split" | "ndarray.array.array_split" | "ndarray.array.split_at" | "ndarray.array.count_nonzero" | "ndarray.array.count_nonzero_axes")
}

impl Worker {
    pub(super) fn execute_chunk_6(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.matrix_rank" => {
                let [Arg::Handle(array), Arg::Float(tolerance)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "matrix_rank expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.matrix_rank(*tolerance) { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.to_npy" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "to_npy expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.to_npy() { Ok(value) if value.is_empty() => "ok_bytes".to_string(), Ok(value) => format!("ok_bytes {}", STANDARD.encode(value)), Err(error) => native_error(&error) }
            }
            "ndarray.array.from_npy" => {
                let [Arg::Bytes(bytes), Arg::Int(maximum)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_npy expects its declared arguments");
                };
                let value = match Array::from_npy(bytes.as_slice(), *maximum) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.save_npy" => {
                let [Arg::Handle(array), Arg::String(path)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "save_npy expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.save_npy(path.as_str()) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.load_npy" => {
                let [Arg::String(path), Arg::Int(maximum)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "load_npy expects its declared arguments");
                };
                let value = match Array::load_npy(path.as_str(), *maximum) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.serialize" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "serialize expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.serialize() { Ok(value) if value.is_empty() => "ok_bytes".to_string(), Ok(value) => format!("ok_bytes {}", STANDARD.encode(value)), Err(error) => native_error(&error) }
            }
            "ndarray.array.deserialize" => {
                let [Arg::Bytes(bytes), Arg::Int(maximum)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "deserialize expects its declared arguments");
                };
                let value = match Array::deserialize(bytes.as_slice(), *maximum) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.mmap_npy" => {
                let [Arg::String(path), Arg::Bool(writable), Arg::Int(maximum)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "mmap_npy expects its declared arguments");
                };
                let value = match Array::mmap_npy(path.as_str(), *writable, *maximum) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.flush" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "flush expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.flush() { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.to_arrow" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "to_arrow expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.to_arrow() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::ArrowArray(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.ArrowArray.ArrowArray"))
            }
            "ndarray.array.from_arrow" => {
                let [Arg::Handle(arrow)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_arrow expects its declared arguments");
                };
                let value_arrow = match self.live_arrowarray(arrow) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match Array::from_arrow(value_arrow) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.split" => {
                let [Arg::Handle(array), Arg::Int(sections), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "split expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.split(*sections, *axis) {
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
            "ndarray.array.array_split" => {
                let [Arg::Handle(array), Arg::Int(sections), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "array_split expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.array_split(*sections, *axis) {
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
            "ndarray.array.split_at" => {
                let [Arg::Handle(array), indices @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "split_at expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.split_at(arg_ints(indices), *axis) {
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
            "ndarray.array.count_nonzero" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "count_nonzero expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.count_nonzero() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.count_nonzero_axes" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "count_nonzero_axes expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.count_nonzero_axes(arg_ints(axes), *keep_dims) {
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
