use super::*;

pub(super) fn accepts_chunk_45(operation: &str) -> bool {
    matches!(operation, "ndarray.numpy.hstack" | "ndarray.numpy.insert" | "ndarray.numpy.insert_indexes" | "ndarray.numpy.insert_scalar_axis" | "ndarray.numpy.insert_indexes_axis" | "ndarray.numpy.ix_" | "ndarray.numpy.may_share_memory" | "ndarray.numpy.may_share_memory_with_work" | "ndarray.numpy.internal_overlap" | "ndarray.numpy.meshgrid_native" | "ndarray.numpy.msort" | "ndarray.numpy.pad" | "ndarray.numpy.place" | "ndarray.numpy.putmask" | "ndarray.numpy.ravel_multi_index_native" | "ndarray.numpy.row_stack")
}

impl Worker {
    pub(super) fn execute_chunk_45(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.numpy.hstack" => {
                let [arrays @ (Arg::Handles(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "hstack expects its declared arguments");
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
                let value = match hstack(value_arrays.as_slice()) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.insert" => {
                let [Arg::Handle(array), Arg::Int(index), Arg::Handle(values)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "insert expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_values = match self.live_array(values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.insert(*index, value_values) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.insert_indexes" => {
                let [Arg::Handle(array), indexes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Handle(values)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "insert expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_values = match self.live_array(values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.insert_indexes(arg_ints(indexes), value_values) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.insert_scalar_axis" => {
                let [Arg::Handle(array), Arg::Int(index), Arg::Handle(values), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "insert expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_values = match self.live_array(values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.insert_scalar_axis(*index, value_values, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.insert_indexes_axis" => {
                let [Arg::Handle(array), indexes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Handle(values), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "insert expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_values = match self.live_array(values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.insert_indexes_axis(arg_ints(indexes), value_values, *axis) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.ix_" => {
                let [Arg::Handle(first), remaining @ (Arg::Handles(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "ix_ expects its declared arguments");
                };
                let value_first = match self.live_array(first) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_remaining = {
                    let mut values = Vec::new();
                    for handle in arg_handles(remaining) {
                        let value = match self.live_array(handle) {
                            Ok(value) => value,
                            Err(error) => return error,
                        };
                        values.push(value);
                    }
                    values
                };
                let values = match value_first.ix_(value_remaining.as_slice()) {
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
            "ndarray.numpy.may_share_memory" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "may_share_memory expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_lhs.may_share_memory(value_rhs) { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.may_share_memory_with_work" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs), Arg::Int(max_work)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "may_share_memory_with_work expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_lhs.may_share_memory_with_work(value_rhs, *max_work) { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.internal_overlap" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "internal_overlap expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.internal_overlap() { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.meshgrid_native" => {
                let [Arg::Handle(first), remaining @ (Arg::Handles(_) | Arg::EmptyList), Arg::Int(indexing), Arg::Bool(sparse), Arg::Bool(copy)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "meshgrid_native expects its declared arguments");
                };
                let value_first = match self.live_array(first) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_remaining = {
                    let mut values = Vec::new();
                    for handle in arg_handles(remaining) {
                        let value = match self.live_array(handle) {
                            Ok(value) => value,
                            Err(error) => return error,
                        };
                        values.push(value);
                    }
                    values
                };
                let values = match value_first.meshgrid_native(value_remaining.as_slice(), *indexing, *sparse, *copy) {
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
            "ndarray.numpy.msort" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "msort expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.msort() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.pad" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.pad(arg_ints(leading), arg_ints(trailing), *value) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.place" => {
                let [Arg::Handle(array), Arg::Handle(mask), Arg::Handle(values)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "place expects its declared arguments");
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
                match value_array.place(value_mask, value_values) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.putmask" => {
                let [Arg::Handle(array), Arg::Handle(mask), Arg::Handle(values)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "putmask expects its declared arguments");
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
                match value_array.putmask(value_mask, value_values) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.ravel_multi_index_native" => {
                let [Arg::Handle(coordinates), dimensions @ (Arg::Ints(_) | Arg::EmptyList), modes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "ravel_multi_index_native expects its declared arguments");
                };
                let value_coordinates = match self.live_array(coordinates) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_coordinates.ravel_multi_index_native(arg_ints(dimensions), arg_ints(modes), *order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.row_stack" => {
                let [arrays @ (Arg::Handles(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "row_stack expects its declared arguments");
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
                let value = match row_stack(value_arrays.as_slice()) {
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
