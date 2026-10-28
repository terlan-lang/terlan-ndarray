use super::*;

pub(super) fn accepts_chunk_19(operation: &str) -> bool {
    matches!(operation, "ndarray.array.in1d" | "ndarray.array.set_difference_unique" | "ndarray.array.intersect_all" | "ndarray.array.unique_axis_all" | "ndarray.array.ediff1d_edges" | "ndarray.array.cumulative_sum" | "ndarray.array.cumulative_product" | "ndarray.array.export_tensor_packet" | "ndarray.array.import_tensor_packet" | "ndarray.array.choose" | "ndarray.array.fill" | "ndarray.array.resize" | "ndarray.array.any_where" | "ndarray.array.all_where" | "ndarray.array.choose_into" | "ndarray.array.round_decimals")
}

impl Worker {
    pub(super) fn execute_chunk_19(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.in1d" => {
                let [Arg::Handle(elements), Arg::Handle(test_elements), Arg::Bool(invert)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "in1d expects its declared arguments");
                };
                let value_elements = match self.live_array(elements) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_test_elements = match self.live_array(test_elements) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_elements.in1d(value_test_elements, *invert) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.set_difference_unique" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "set_difference_unique expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.set_difference_unique(value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.intersect_all" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "intersect_all expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_lhs.intersect_all(value_rhs) {
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
            "ndarray.array.unique_axis_all" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "unique_axis_all expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.unique_axis_all(*axis) {
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
            "ndarray.array.ediff1d_edges" => {
                let [Arg::Handle(array), Arg::Handle(begin), Arg::Handle(end)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "ediff1d_edges expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_begin = match self.live_array(begin) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_end = match self.live_array(end) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.ediff1d_edges(value_begin, value_end) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.cumulative_sum" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "cumulative_sum expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.cumulative_sum(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.cumulative_product" => {
                let [Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "cumulative_product expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.cumulative_product(*axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.export_tensor_packet" => {
                let [Arg::Handle(array), Arg::String(consumer)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "export_tensor_packet expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.export_tensor_packet(consumer.as_str()) { Ok(value) if value.is_empty() => "ok_bytes".to_string(), Ok(value) => format!("ok_bytes {}", STANDARD.encode(value)), Err(error) => native_error(&error) }
            }
            "ndarray.array.import_tensor_packet" => {
                let [Arg::Bytes(packet)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "import_tensor_packet expects its declared arguments");
                };
                let value = match Array::import_tensor_packet(packet.as_slice()) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.choose" => {
                let [Arg::Handle(indexes), choices @ (Arg::Handles(_) | Arg::EmptyList), Arg::Int(mode)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "choose expects its declared arguments");
                };
                let value_indexes = match self.live_array(indexes) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_choices = {
                    let mut values = Vec::new();
                    for handle in arg_handles(choices) {
                        let value = match self.live_array(handle) {
                            Ok(value) => value,
                            Err(error) => return error,
                        };
                        values.push(value);
                    }
                    values
                };
                let value = match value_indexes.choose(value_choices.as_slice(), *mode) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.fill" => {
                let [Arg::Handle(array), Arg::Handle(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "fill expects its declared arguments");
                };
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(value, "ndarray.Array.Array") {
                    return error;
                }
                if array.id == value.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_array, entry_value] = self.handles.get_disjoint_mut([&array.id, &value.id]);
                let value_array = match &mut entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_value = match &entry_value.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_array.fill(value_value) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.resize" => {
                let [Arg::Handle(array), shape @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "resize expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.resize(arg_ints(shape)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.any_where" => {
                let [Arg::Handle(array), Arg::Handle(mask), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "any_where expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_mask = match self.live_array(mask) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.any_where(value_mask, arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.all_where" => {
                let [Arg::Handle(array), Arg::Handle(mask), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "all_where expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_mask = match self.live_array(mask) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.all_where(value_mask, arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.choose_into" => {
                let [Arg::Handle(destination), Arg::Handle(indexes), choices @ (Arg::Handles(_) | Arg::EmptyList), Arg::Int(mode)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "choose_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(indexes, "ndarray.Array.Array") {
                    return error;
                }
                if destination.id == indexes.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require the receiver to be distinct from borrowed resources");
                }
                for handle in arg_handles(choices) {
                    if let Err(error) = self.validate(handle, "ndarray.Array.Array") {
                        return error;
                    }
                    if handle.id == destination.id {
                        return protocol_error("aliased_mutable_handle", "mutable resource calls cannot borrow the receiver through a resource list");
                    }
                }
                let mut entry_destination = self.handles.remove(&destination.id).expect("validated mutable receiver");
                let value_destination = match &mut entry_destination.value {
                    HandleValue::Array(value) => value,
                    _ => unreachable!("validated mutable receiver type"),
                };
                let value_indexes = match &self.handles.get(&indexes.id).expect("validated borrowed resource").value {
                    HandleValue::Array(value) => value,
                    _ => unreachable!("validated borrowed resource type"),
                };
                let value_choices = arg_handles(choices).iter().map(|handle| match &self.handles.get(&handle.id).expect("validated borrowed list resource").value {
                    HandleValue::Array(value) => value,
                    _ => unreachable!("validated borrowed list resource type"),
                }).collect::<Vec<_>>();
                let call_result = value_destination.choose_into(value_indexes, value_choices.as_slice(), *mode);
                let previous = self.handles.insert(destination.id, entry_destination);
                debug_assert!(previous.is_none());
                match call_result { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.round_decimals" => {
                let [Arg::Handle(array), Arg::Int(decimals)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "round_decimals expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.round_decimals(*decimals) {
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
