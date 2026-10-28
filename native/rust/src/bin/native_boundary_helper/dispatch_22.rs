use super::*;

pub(super) fn accepts_chunk_22(operation: &str) -> bool {
    matches!(operation, "ndarray.creation.full_like_int" | "ndarray.creation.full_like_float" | "ndarray.creation.logspace" | "ndarray.creation.geomspace" | "ndarray.creation.logspace_arrays" | "ndarray.creation.geomspace_arrays" | "ndarray.creation.tri" | "ndarray.creation.tril" | "ndarray.creation.triu" | "ndarray.creation.diagflat" | "ndarray.creation.indices" | "ndarray.creation.indices_sparse" | "ndarray.creation.vander" | "ndarray.shape.atleast_1d" | "ndarray.shape.atleast_2d" | "ndarray.shape.atleast_3d")
}

impl Worker {
    pub(super) fn execute_chunk_22(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.creation.full_like_int" => {
                let [Arg::Handle(array), Arg::Int(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "full_like_int expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match full_like_int(value_array, *value) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.creation.full_like_float" => {
                let [Arg::Handle(array), Arg::Float(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "full_like_float expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match full_like_float(value_array, *value) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.creation.logspace" => {
                let [Arg::Float(start), Arg::Float(stop), Arg::Int(count), Arg::Int(endpoint), Arg::Float(base), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "logspace expects its declared arguments");
                };
                let value = match logspace(*start, *stop, *count, *endpoint, *base, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.creation.geomspace" => {
                let [Arg::Float(start), Arg::Float(stop), Arg::Int(count), Arg::Int(endpoint), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "geomspace expects its declared arguments");
                };
                let value = match geomspace(*start, *stop, *count, *endpoint, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.creation.logspace_arrays" => {
                let [Arg::Handle(start), Arg::Handle(stop), Arg::Int(count), Arg::Int(endpoint), Arg::Int(axis), Arg::Float(base), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "logspace_arrays expects its declared arguments");
                };
                let value_start = match self.live_array(start) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_stop = match self.live_array(stop) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match logspace_arrays(value_start, value_stop, *count, *endpoint, *axis, *base, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.creation.geomspace_arrays" => {
                let [Arg::Handle(start), Arg::Handle(stop), Arg::Int(count), Arg::Int(endpoint), Arg::Int(axis), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "geomspace_arrays expects its declared arguments");
                };
                let value_start = match self.live_array(start) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_stop = match self.live_array(stop) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match geomspace_arrays(value_start, value_stop, *count, *endpoint, *axis, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.creation.tri" => {
                let [Arg::Int(rows), Arg::Int(columns), Arg::Int(diagonal), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "tri expects its declared arguments");
                };
                let value = match tri(*rows, *columns, *diagonal, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.creation.tril" => {
                let [Arg::Handle(array), Arg::Int(diagonal)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "tril expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match tril(value_array, *diagonal) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.creation.triu" => {
                let [Arg::Handle(array), Arg::Int(diagonal)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "triu expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match triu(value_array, *diagonal) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.creation.diagflat" => {
                let [Arg::Handle(array), Arg::Int(diagonal)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "diagflat expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match diagflat(value_array, *diagonal) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.creation.indices" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "indices expects its declared arguments");
                };
                let value = match indices(arg_ints(shape), *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.creation.indices_sparse" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "indices_sparse expects its declared arguments");
                };
                let values = match indices_sparse(arg_ints(shape), *dtype) {
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
            "ndarray.creation.vander" => {
                let [Arg::Handle(array), Arg::Int(columns), Arg::Int(increasing)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "vander expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match vander(value_array, *columns, *increasing) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.atleast_1d" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "atleast_1d expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match atleast_1d(value_array) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.atleast_2d" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "atleast_2d expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match atleast_2d(value_array) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.atleast_3d" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "atleast_3d expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match atleast_3d(value_array) {
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
