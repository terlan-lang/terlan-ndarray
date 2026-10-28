use super::*;

pub(super) fn accepts_chunk_4(operation: &str) -> bool {
    matches!(operation, "ndarray.array.put_along_axis" | "ndarray.array.compress" | "ndarray.array.compress_flat" | "ndarray.array.mask_select" | "ndarray.array.mask_assign" | "ndarray.array.gather_coordinates" | "ndarray.array.put_coordinates" | "ndarray.array.item_bool" | "ndarray.array.item_int" | "ndarray.array.get_bool" | "ndarray.array.get_int" | "ndarray.array.get_float" | "ndarray.array.set_bool" | "ndarray.array.set_int" | "ndarray.array.set_float" | "ndarray.array.add")
}

impl Worker {
    pub(super) fn execute_chunk_4(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.put_along_axis" => {
                let [Arg::Handle(array), Arg::Handle(indices), Arg::Handle(values), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "put_along_axis expects its declared arguments");
                };
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(indices, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(values, "ndarray.Array.Array") {
                    return error;
                }
                if array.id == indices.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if array.id == values.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if indices.id == values.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_array, entry_indices, entry_values] = self.handles.get_disjoint_mut([&array.id, &indices.id, &values.id]);
                let value_array = match &mut entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_indices = match &entry_indices.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_values = match &entry_values.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_array.put_along_axis(value_indices, value_values, *axis) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.compress" => {
                let [Arg::Handle(array), Arg::Handle(mask), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "compress expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_mask = match self.live_array(mask) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.compress(value_mask, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.compress_flat" => {
                let [Arg::Handle(array), Arg::Handle(mask)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "compress_flat expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_mask = match self.live_array(mask) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.compress_flat(value_mask) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.mask_select" => {
                let [Arg::Handle(array), Arg::Handle(mask)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "mask_select expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_mask = match self.live_array(mask) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.mask_select(value_mask) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.mask_assign" => {
                let [Arg::Handle(array), Arg::Handle(mask), Arg::Handle(values)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "mask_assign expects its declared arguments");
                };
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(mask, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(values, "ndarray.Array.Array") {
                    return error;
                }
                if array.id == mask.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if array.id == values.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if mask.id == values.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_array, entry_mask, entry_values] = self.handles.get_disjoint_mut([&array.id, &mask.id, &values.id]);
                let value_array = match &mut entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_mask = match &entry_mask.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_values = match &entry_values.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_array.mask_assign(value_mask, value_values) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.gather_coordinates" => {
                let [Arg::Handle(array), Arg::Handle(coordinates)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "gather_coordinates expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.gather_coordinates(value_coordinates) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.put_coordinates" => {
                let [Arg::Handle(array), Arg::Handle(coordinates), Arg::Handle(values)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "put_coordinates expects its declared arguments");
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
                match value_array.put_coordinates(value_coordinates, value_values) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.item_bool" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "item_bool expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.item_bool() { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.item_int" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "item_int expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.item_int() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.get_bool" => {
                let [Arg::Handle(array), indices @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "get_bool expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.get_bool(arg_ints(indices)) { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.get_int" => {
                let [Arg::Handle(array), indices @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "get_int expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.get_int(arg_ints(indices)) { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.get_float" => {
                let [Arg::Handle(array), indices @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "get_float expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.get_float(arg_ints(indices)) { Ok(value) => format!("ok_float {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.set_bool" => {
                let [Arg::Handle(array), indices @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "set_bool expects its declared arguments");
                };
                let value_array = match self.live_array_mut(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.set_bool(arg_ints(indices), *value) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.set_int" => {
                let [Arg::Handle(array), indices @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "set_int expects its declared arguments");
                };
                let value_array = match self.live_array_mut(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.set_int(arg_ints(indices), *value) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.set_float" => {
                let [Arg::Handle(array), indices @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "set_float expects its declared arguments");
                };
                let value_array = match self.live_array_mut(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.set_float(arg_ints(indices), *value) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.add" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "add expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.add(value_rhs) {
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
