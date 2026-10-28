use super::*;

pub(super) fn accepts_chunk_9(operation: &str) -> bool {
    matches!(operation, "ndarray.array.partition_inplace" | "ndarray.array.partition_kths_inplace" | "ndarray.array.broadcast_to" | "ndarray.array.true_divide" | "ndarray.array.floor_divide" | "ndarray.array.remainder" | "ndarray.array.power" | "ndarray.array.elementwise_minimum" | "ndarray.array.elementwise_maximum" | "ndarray.array.fmin" | "ndarray.array.fmax" | "ndarray.array.clip" | "ndarray.array.clip_lower" | "ndarray.array.clip_upper" | "ndarray.array.clip_into" | "ndarray.array.clip_lower_into")
}

impl Worker {
    pub(super) fn execute_chunk_9(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.partition_inplace" => {
                let [Arg::Handle(array), Arg::Int(kth), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "partition_inplace expects its declared arguments");
                };
                let value_array = match self.live_array_mut(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.partition_inplace(*kth, *axis) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.partition_kths_inplace" => {
                let [Arg::Handle(array), Arg::Handle(kths), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "partition_inplace expects its declared arguments");
                };
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(kths, "ndarray.Array.Array") {
                    return error;
                }
                if array.id == kths.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_array, entry_kths] = self.handles.get_disjoint_mut([&array.id, &kths.id]);
                let value_array = match &mut entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_kths = match &entry_kths.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_array.partition_kths_inplace(value_kths, *axis) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.broadcast_to" => {
                let [Arg::Handle(array), shape @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "broadcast_to expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.broadcast_to(arg_ints(shape)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.true_divide" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "true_divide expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.true_divide(value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.floor_divide" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "floor_divide expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.floor_divide(value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.remainder" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "remainder expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.remainder(value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.power" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "power expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.power(value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.elementwise_minimum" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "elementwise_minimum expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.elementwise_minimum(value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.elementwise_maximum" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "elementwise_maximum expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.elementwise_maximum(value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.fmin" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "fmin expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.fmin(value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.fmax" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "fmax expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_lhs.fmax(value_rhs) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.clip" => {
                let [Arg::Handle(array), Arg::Handle(minimum), Arg::Handle(maximum)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "clip expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_minimum = match self.live_array(minimum) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_maximum = match self.live_array(maximum) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.clip(value_minimum, value_maximum) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.clip_lower" => {
                let [Arg::Handle(array), Arg::Handle(minimum)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "clip_lower expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_minimum = match self.live_array(minimum) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.clip_lower(value_minimum) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.clip_upper" => {
                let [Arg::Handle(array), Arg::Handle(maximum)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "clip_upper expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_maximum = match self.live_array(maximum) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.clip_upper(value_maximum) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.clip_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Handle(minimum), Arg::Handle(maximum)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "clip_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(minimum, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(maximum, "ndarray.Array.Array") {
                    return error;
                }
                if destination.id == array.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if destination.id == minimum.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if destination.id == maximum.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if array.id == minimum.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if array.id == maximum.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if minimum.id == maximum.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_destination, entry_array, entry_minimum, entry_maximum] = self.handles.get_disjoint_mut([&destination.id, &array.id, &minimum.id, &maximum.id]);
                let value_destination = match &mut entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_array = match &entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_minimum = match &entry_minimum.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_maximum = match &entry_maximum.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_destination.clip_into(value_array, value_minimum, value_maximum) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.array.clip_lower_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Handle(minimum)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "clip_lower_into expects its declared arguments");
                };
                if let Err(error) = self.validate(destination, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                if let Err(error) = self.validate(minimum, "ndarray.Array.Array") {
                    return error;
                }
                if destination.id == array.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if destination.id == minimum.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                if array.id == minimum.id {
                    return protocol_error("aliased_mutable_handle", "mutable resource calls require distinct handle arguments");
                }
                let [entry_destination, entry_array, entry_minimum] = self.handles.get_disjoint_mut([&destination.id, &array.id, &minimum.id]);
                let value_destination = match &mut entry_destination.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_array = match &entry_array.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                let value_minimum = match &entry_minimum.expect("validated disjoint handle").value {
                    HandleValue::Array(value) => value,
                    _ => return protocol_error("handle_storage_mismatch", "ndarray.Array.Array"),
                };
                match value_destination.clip_lower_into(value_array, value_minimum) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            _ => unreachable!("operation routed to the wrong generated dispatch chunk"),
        }
    }
}
