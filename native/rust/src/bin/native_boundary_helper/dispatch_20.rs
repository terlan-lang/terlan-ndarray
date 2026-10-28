use super::*;

pub(super) fn accepts_chunk_20(operation: &str) -> bool {
    matches!(operation, "ndarray.array.round_into" | "ndarray.array.conjugate_into" | "ndarray.array.matmul_into" | "ndarray.array.dot_into" | "ndarray.array.trace_into" | "ndarray.array.dispose" | "ndarray.array.outer_into" | "ndarray.array.sum_into" | "ndarray.array.product_into" | "ndarray.array.minimum_into" | "ndarray.array.maximum_into" | "ndarray.array.sum_where" | "ndarray.array.sum_initial" | "ndarray.array.sum_where_initial" | "ndarray.array.minimum_initial" | "ndarray.array.maximum_initial")
}

impl Worker {
    pub(super) fn execute_chunk_20(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.round_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Int(decimals)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "round_into expects its declared arguments");
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
                match value_destination.round_into(value_array, *decimals) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.conjugate_into" => {
                let [Arg::Handle(destination), Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "conjugate_into expects its declared arguments");
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
                match value_destination.conjugate_into(value_array) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.matmul_into" => {
                let [Arg::Handle(destination), Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "matmul_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(lhs, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(rhs, "ndarray.Array.Array") {
                    return error;
                }
                if destination.id == lhs.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if destination.id == rhs.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if lhs.id == rhs.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_destination, entry_lhs, entry_rhs] = self.handles.get_disjoint_mut([&destination.id, &lhs.id, &rhs.id]);
                let value_destination = match &mut entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_lhs = match &entry_lhs.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_rhs = match &entry_rhs.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_destination.matmul_into(value_lhs, value_rhs) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.dot_into" => {
                let [Arg::Handle(destination), Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dot_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(lhs, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(rhs, "ndarray.Array.Array") {
                    return error;
                }
                if destination.id == lhs.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if destination.id == rhs.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if lhs.id == rhs.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_destination, entry_lhs, entry_rhs] = self.handles.get_disjoint_mut([&destination.id, &lhs.id, &rhs.id]);
                let value_destination = match &mut entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_lhs = match &entry_lhs.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_rhs = match &entry_rhs.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_destination.dot_into(value_lhs, value_rhs) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.trace_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Int(offset), Arg::Int(axis_a), Arg::Int(axis_b)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "trace_into expects its declared arguments");
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
                match value_destination.trace_into(value_array, *offset, *axis_a, *axis_b) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.dispose" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dispose expects its declared arguments");
                };
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                self.release_handle(array.id);
                "ok_unit".to_string()
            }
            "ndarray.array.outer_into" => {
                let [Arg::Handle(destination), Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "outer_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(lhs, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(rhs, "ndarray.Array.Array") {
                    return error;
                }
                if destination.id == lhs.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if destination.id == rhs.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if lhs.id == rhs.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_destination, entry_lhs, entry_rhs] = self.handles.get_disjoint_mut([&destination.id, &lhs.id, &rhs.id]);
                let value_destination = match &mut entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_lhs = match &entry_lhs.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_rhs = match &entry_rhs.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_destination.outer_into(value_lhs, value_rhs) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.sum_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sum_into expects its declared arguments");
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
                match value_destination.sum_into(value_array, arg_ints(axes), *keep_dims) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.product_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "product_into expects its declared arguments");
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
                match value_destination.product_into(value_array, arg_ints(axes), *keep_dims) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.minimum_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "minimum_into expects its declared arguments");
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
                match value_destination.minimum_into(value_array, arg_ints(axes), *keep_dims) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.maximum_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "maximum_into expects its declared arguments");
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
                match value_destination.maximum_into(value_array, arg_ints(axes), *keep_dims) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.sum_where" => {
                let [Arg::Handle(array), Arg::Handle(mask), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sum_where expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_mask = match self.live_array(mask) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.sum_where(value_mask, arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.sum_initial" => {
                let [Arg::Handle(array), Arg::Handle(initial), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sum_initial expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_initial = match self.live_array(initial) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.sum_initial(value_initial, arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.sum_where_initial" => {
                let [Arg::Handle(array), Arg::Handle(mask), Arg::Handle(initial), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "sum_where_initial expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_mask = match self.live_array(mask) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_initial = match self.live_array(initial) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.sum_where_initial(value_mask, value_initial, arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.minimum_initial" => {
                let [Arg::Handle(array), Arg::Handle(initial), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "minimum_initial expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_initial = match self.live_array(initial) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.minimum_initial(value_initial, arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.maximum_initial" => {
                let [Arg::Handle(array), Arg::Handle(initial), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "maximum_initial expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_initial = match self.live_array(initial) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.maximum_initial(value_initial, arg_ints(axes), *keep_dims) {
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
