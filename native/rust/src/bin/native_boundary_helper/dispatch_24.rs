use super::*;

pub(super) fn accepts_chunk_24(operation: &str) -> bool {
    matches!(operation, "ndarray.indexing.take_indices" | "ndarray.indexing.take_indices_mode" | "ndarray.indexing.take_indices_into" | "ndarray.indexing.compress_into" | "ndarray.indexing.compress_flat_into" | "ndarray.indexing.add_at_coordinates" | "ndarray.shape.pad_constant" | "ndarray.shape.pad_edge" | "ndarray.shape.pad_reflect" | "ndarray.shape.pad_symmetric" | "ndarray.shape.pad_wrap" | "ndarray.shape.pad_reflect_odd" | "ndarray.shape.pad_symmetric_odd" | "ndarray.shape.pad_linear_ramp" | "ndarray.shape.pad_empty" | "ndarray.shape.pad_minimum")
}

impl Worker {
    pub(super) fn execute_chunk_24(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.indexing.take_indices" => {
                let [Arg::Handle(array), Arg::Handle(indices), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "take_indices expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_indices = match self.live_array(indices) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match take_indices(value_array, value_indices, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.indexing.take_indices_mode" => {
                let [Arg::Handle(array), Arg::Handle(indices), Arg::Int(axis), Arg::Int(mode)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "take_indices_mode expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_indices = match self.live_array(indices) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match take_indices_mode(value_array, value_indices, *axis, *mode) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.indexing.take_indices_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Handle(indices), Arg::Int(axis), Arg::Int(mode)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "take_indices_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(indices, "ndarray.Array.Array") {
                    return error;
                }
                if destination.id == array.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if destination.id == indices.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if array.id == indices.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_destination, entry_array, entry_indices] = self.handles.get_disjoint_mut([&destination.id, &array.id, &indices.id]);
                let value_destination = match &mut entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_array = match &entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_indices = match &entry_indices.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_destination.take_indices_into(value_array, value_indices, *axis, *mode) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.indexing.compress_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Handle(mask), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "compress_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(mask, "ndarray.Array.Array") {
                    return error;
                }
                if destination.id == array.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if destination.id == mask.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if array.id == mask.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_destination, entry_array, entry_mask] = self.handles.get_disjoint_mut([&destination.id, &array.id, &mask.id]);
                let value_destination = match &mut entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_array = match &entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_mask = match &entry_mask.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_destination.compress_into(value_array, value_mask, *axis) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.indexing.compress_flat_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Handle(mask)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "compress_flat_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(mask, "ndarray.Array.Array") {
                    return error;
                }
                if destination.id == array.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if destination.id == mask.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if array.id == mask.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_destination, entry_array, entry_mask] = self.handles.get_disjoint_mut([&destination.id, &array.id, &mask.id]);
                let value_destination = match &mut entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_array = match &entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_mask = match &entry_mask.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_destination.compress_flat_into(value_array, value_mask) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.indexing.add_at_coordinates" => {
                let [Arg::Handle(array), Arg::Handle(coordinates), Arg::Handle(values)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "add_at_coordinates expects its declared arguments");
                };
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(coordinates, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(values, "ndarray.Array.Array") {
                    return error;
                }
                if array.id == coordinates.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if array.id == values.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if coordinates.id == values.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_array, entry_coordinates, entry_values] = self.handles.get_disjoint_mut([&array.id, &coordinates.id, &values.id]);
                let value_array = match &mut entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_coordinates = match &entry_coordinates.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_values = match &entry_values.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_array.add_at_coordinates(value_coordinates, value_values) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.shape.pad_constant" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList), leading_values @ (Arg::Floats(_) | Arg::EmptyList), trailing_values @ (Arg::Floats(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_constant expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_constant(value_array, arg_ints(leading), arg_ints(trailing), arg_floats(leading_values), arg_floats(trailing_values)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_edge" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_edge expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_edge(value_array, arg_ints(leading), arg_ints(trailing)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_reflect" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_reflect expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_reflect(value_array, arg_ints(leading), arg_ints(trailing)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_symmetric" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_symmetric expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_symmetric(value_array, arg_ints(leading), arg_ints(trailing)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_wrap" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_wrap expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_wrap(value_array, arg_ints(leading), arg_ints(trailing)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_reflect_odd" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_reflect_odd expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_reflect_odd(value_array, arg_ints(leading), arg_ints(trailing)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_symmetric_odd" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_symmetric_odd expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_symmetric_odd(value_array, arg_ints(leading), arg_ints(trailing)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_linear_ramp" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList), leading_values @ (Arg::Floats(_) | Arg::EmptyList), trailing_values @ (Arg::Floats(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_linear_ramp expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_linear_ramp(value_array, arg_ints(leading), arg_ints(trailing), arg_floats(leading_values), arg_floats(trailing_values)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_empty" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_empty expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_empty(value_array, arg_ints(leading), arg_ints(trailing)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_minimum" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList), leading_lengths @ (Arg::Ints(_) | Arg::EmptyList), trailing_lengths @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_minimum expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_minimum(value_array, arg_ints(leading), arg_ints(trailing), arg_ints(leading_lengths), arg_ints(trailing_lengths)) {
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
