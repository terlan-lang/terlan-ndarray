use super::*;

pub(super) fn accepts_chunk_23(operation: &str) -> bool {
    matches!(operation, "ndarray.shape.ravel" | "ndarray.shape.ravel_ordered" | "ndarray.shape.flatten" | "ndarray.shape.flatten_ordered" | "ndarray.shape.moveaxis" | "ndarray.shape.rollaxis" | "ndarray.shape.flip_all" | "ndarray.shape.flipud" | "ndarray.shape.fliplr" | "ndarray.shape.rot90" | "ndarray.shape.concatenate_flat" | "ndarray.shape.concatenate_with_dtype" | "ndarray.shape.concatenate_flat_with_dtype" | "ndarray.shape.concatenate_into" | "ndarray.shape.concatenate_flat_into" | "ndarray.shape.basic_slice")
}

impl Worker {
    pub(super) fn execute_chunk_23(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.shape.ravel" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "ravel expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match ravel(value_array) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.ravel_ordered" => {
                let [Arg::Handle(array), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "ravel expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match ravel_ordered(value_array, *order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.flatten" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "flatten expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match flatten(value_array) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.flatten_ordered" => {
                let [Arg::Handle(array), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "flatten expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match flatten_ordered(value_array, *order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.moveaxis" => {
                let [Arg::Handle(array), sources @ (Arg::Ints(_) | Arg::EmptyList), destinations @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "moveaxis expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match moveaxis(value_array, arg_ints(sources), arg_ints(destinations)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.rollaxis" => {
                let [Arg::Handle(array), Arg::Int(axis), Arg::Int(start)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "rollaxis expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match rollaxis(value_array, *axis, *start) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.flip_all" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "flip_all expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match flip_all(value_array) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.flipud" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "flipud expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match flipud(value_array) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.fliplr" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "fliplr expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match fliplr(value_array) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.rot90" => {
                let [Arg::Handle(array), Arg::Int(rotations), Arg::Int(axis_a), Arg::Int(axis_b)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "rot90 expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match rot90(value_array, *rotations, *axis_a, *axis_b) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.concatenate_flat" => {
                let [arrays @ (Arg::Handles(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "concatenate_flat expects its declared arguments");
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
                let value = match concatenate_flat(value_arrays.as_slice()) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.concatenate_with_dtype" => {
                let [arrays @ (Arg::Handles(_) | Arg::EmptyList), Arg::Int(axis), Arg::Int(dtype), Arg::Int(casting)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "concatenate_with_dtype expects its declared arguments");
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
                let value = match concatenate_with_dtype(value_arrays.as_slice(), *axis, *dtype, *casting) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.concatenate_flat_with_dtype" => {
                let [arrays @ (Arg::Handles(_) | Arg::EmptyList), Arg::Int(dtype), Arg::Int(casting)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "concatenate_flat_with_dtype expects its declared arguments");
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
                let value = match concatenate_flat_with_dtype(value_arrays.as_slice(), *dtype, *casting) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.concatenate_into" => {
                let [arrays @ (Arg::Handles(_) | Arg::EmptyList), Arg::Int(axis), Arg::Handle(destination), Arg::Int(casting)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "concatenate_into expects its declared arguments");
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
                let value_destination = match self.live_array(destination) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match concatenate_into(value_arrays.as_slice(), *axis, value_destination, *casting) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.shape.concatenate_flat_into" => {
                let [arrays @ (Arg::Handles(_) | Arg::EmptyList), Arg::Handle(destination), Arg::Int(casting)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "concatenate_flat_into expects its declared arguments");
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
                let value_destination = match self.live_array(destination) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match concatenate_flat_into(value_arrays.as_slice(), value_destination, *casting) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.shape.basic_slice" => {
                let [Arg::Handle(array), starts @ (Arg::Ints(_) | Arg::EmptyList), stops @ (Arg::Ints(_) | Arg::EmptyList), steps @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(ellipsis_index), squeeze_axes @ (Arg::Ints(_) | Arg::EmptyList), new_axes @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "basic_slice expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match basic_slice(value_array, arg_ints(starts), arg_ints(stops), arg_ints(steps), *ellipsis_index, arg_ints(squeeze_axes), arg_ints(new_axes)) {
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
