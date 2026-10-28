use super::*;

pub(super) fn accepts_chunk_8(operation: &str) -> bool {
    matches!(operation, "ndarray.array.stack" | "ndarray.array.sort" | "ndarray.array.argsort" | "ndarray.array.sort_unstable" | "ndarray.array.argsort_unstable" | "ndarray.array.sort_flat" | "ndarray.array.argsort_flat" | "ndarray.array.sort_inplace" | "ndarray.array.partition" | "ndarray.array.argpartition" | "ndarray.array.partition_kths" | "ndarray.array.argpartition_kths" | "ndarray.array.partition_flat" | "ndarray.array.argpartition_flat" | "ndarray.array.partition_kths_flat" | "ndarray.array.argpartition_kths_flat")
}

impl Worker {
    pub(super) fn execute_chunk_8(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.stack" => {
                let [arrays @ (Arg::Handles(_) | Arg::EmptyList), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "stack expects its declared arguments");
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
                let value = match stack(value_arrays.as_slice(), *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.sort" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sort expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.sort(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.argsort" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "argsort expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.argsort(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.sort_unstable" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sort_unstable expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.sort_unstable(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.argsort_unstable" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "argsort_unstable expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.argsort_unstable(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.sort_flat" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sort_flat expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.sort_flat() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.argsort_flat" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "argsort_flat expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.argsort_flat() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.sort_inplace" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sort_inplace expects its declared arguments");
                };
                let value_array = match self.live_array_mut(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.sort_inplace(*axis) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.partition" => {
                let [Arg::Handle(array), Arg::Int(kth), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "partition expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.partition(*kth, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.argpartition" => {
                let [Arg::Handle(array), Arg::Int(kth), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "argpartition expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.argpartition(*kth, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.partition_kths" => {
                let [Arg::Handle(array), Arg::Handle(kths), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "partition expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_kths = match self.live_array(kths) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.partition_kths(value_kths, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.argpartition_kths" => {
                let [Arg::Handle(array), Arg::Handle(kths), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "argpartition expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_kths = match self.live_array(kths) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.argpartition_kths(value_kths, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.partition_flat" => {
                let [Arg::Handle(array), Arg::Int(kth)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "partition_flat expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.partition_flat(*kth) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.argpartition_flat" => {
                let [Arg::Handle(array), Arg::Int(kth)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "argpartition_flat expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.argpartition_flat(*kth) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.partition_kths_flat" => {
                let [Arg::Handle(array), Arg::Handle(kths)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "partition_flat expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_kths = match self.live_array(kths) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.partition_kths_flat(value_kths) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.argpartition_kths_flat" => {
                let [Arg::Handle(array), Arg::Handle(kths)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "argpartition_flat expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_kths = match self.live_array(kths) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.argpartition_kths_flat(value_kths) {
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
