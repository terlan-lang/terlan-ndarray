use super::*;

pub(super) fn accepts_chunk_3(operation: &str) -> bool {
    matches!(operation, "ndarray.array.reshape" | "ndarray.array.reshape_ordered" | "ndarray.array.resize_in_place" | "ndarray.array.transpose" | "ndarray.array.copy" | "ndarray.array.copy_ordered" | "ndarray.array.astype" | "ndarray.array.astype_ordered" | "ndarray.array.contiguous" | "ndarray.array.index_axis" | "ndarray.array.slice_axis" | "ndarray.array.permute" | "ndarray.array.take" | "ndarray.array.take_mode" | "ndarray.array.take_along_axis" | "ndarray.array.put")
}

impl Worker {
    pub(super) fn execute_chunk_3(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.reshape" => {
                let [Arg::Handle(array), shape @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "reshape expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.reshape(arg_ints(shape)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.reshape_ordered" => {
                let [Arg::Handle(array), shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "reshape expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.reshape_ordered(arg_ints(shape), *order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.resize_in_place" => {
                let [Arg::Handle(array), shape @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "resize_in_place expects its declared arguments");
                };
                let value_array = match self.live_array_mut(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.resize_in_place(arg_ints(shape)) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.transpose" => {
                let [Arg::Handle(array), Arg::Int(axis_a), Arg::Int(axis_b)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "transpose expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.transpose(*axis_a, *axis_b) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.copy" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "copy expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.copy() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.copy_ordered" => {
                let [Arg::Handle(array), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "copy expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.copy_ordered(*order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.astype" => {
                let [Arg::Handle(array), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "astype expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.astype(*dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.astype_ordered" => {
                let [Arg::Handle(array), Arg::Int(dtype), Arg::Int(order), Arg::Bool(copy), Arg::Int(casting)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "astype expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.astype_ordered(*dtype, *order, *copy, *casting) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.contiguous" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "contiguous expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.contiguous() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.index_axis" => {
                let [Arg::Handle(array), Arg::Int(axis), Arg::Int(index)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "index_axis expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.index_axis(*axis, *index) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.slice_axis" => {
                let [Arg::Handle(array), Arg::Int(axis), Arg::Int(start), Arg::Int(stop), Arg::Int(step)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "slice_axis expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.slice_axis(*axis, *start, *stop, *step) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.permute" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "permute expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.permute(arg_ints(axes)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.take" => {
                let [Arg::Handle(array), indices @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "take expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.take(arg_ints(indices), *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.take_mode" => {
                let [Arg::Handle(array), indices @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(axis), Arg::Int(mode)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "take_mode expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.take_mode(arg_ints(indices), *axis, *mode) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.take_along_axis" => {
                let [Arg::Handle(array), Arg::Handle(indices), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "take_along_axis expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_indices = match self.live_array(indices) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.take_along_axis(value_indices, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.put" => {
                let [Arg::Handle(array), indices @ (Arg::Ints(_) | Arg::EmptyList), Arg::Handle(values)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "put expects its declared arguments");
                };
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(values, "ndarray.Array.Array") {
                    return error;
                }
                if array.id == values.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_array, entry_values] = self.handles.get_disjoint_mut([&array.id, &values.id]);
                let value_array = match &mut entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_values = match &entry_values.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_array.put(arg_ints(indices), value_values) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            _ => unreachable!("operation routed to the wrong generated dispatch chunk"),
        }
    }
}
