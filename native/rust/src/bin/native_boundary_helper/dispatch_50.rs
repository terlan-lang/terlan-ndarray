use super::*;

pub(super) fn accepts_chunk_50(operation: &str) -> bool {
    matches!(operation, "ndarray.numpy.add_at" | "ndarray.numpy.logical_xor_reduce" | "ndarray.numpy.add_reduceat" | "ndarray.numpy.add_reduceat_into")
}

impl Worker {
    pub(super) fn execute_chunk_50(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.numpy.add_at" => {
                let [Arg::Handle(array), Arg::Handle(indexes), Arg::Handle(values)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "add_at expects its declared arguments");
                };
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(indexes, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(values, "ndarray.Array.Array") {
                    return error;
                }
                if array.id == indexes.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if array.id == values.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if indexes.id == values.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_array, entry_indexes, entry_values] = self.handles.get_disjoint_mut([&array.id, &indexes.id, &values.id]);
                let value_array = match &mut entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_indexes = match &entry_indexes.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_values = match &entry_values.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_array.add_at(value_indexes, value_values) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.logical_xor_reduce" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "logical_xor_reduce expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.logical_xor_reduce(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.add_reduceat" => {
                let [Arg::Handle(array), Arg::Handle(indexes), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "add_reduceat expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_indexes = match self.live_array(indexes) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.add_reduceat(value_indexes, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.add_reduceat_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Handle(indexes), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "add_reduceat_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(indexes, "ndarray.Array.Array") {
                    return error;
                }
                if destination.id == array.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if destination.id == indexes.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if array.id == indexes.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_destination, entry_array, entry_indexes] = self.handles.get_disjoint_mut([&destination.id, &array.id, &indexes.id]);
                let value_destination = match &mut entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_array = match &entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_indexes = match &entry_indexes.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_destination.add_reduceat_into(value_array, value_indexes, *axis) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            _ => unreachable!("operation routed to the wrong generated dispatch chunk"),
        }
    }
}
