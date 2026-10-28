use super::*;

pub(super) fn accepts_chunk_48(operation: &str) -> bool {
    matches!(operation, "ndarray.numpy.nanstd_as" | "ndarray.numpy.nanargmin_axes" | "ndarray.numpy.nanargmax_axes" | "ndarray.numpy.nansum_into" | "ndarray.numpy.nanprod_into" | "ndarray.numpy.nanmean_into" | "ndarray.numpy.nanmin_into" | "ndarray.numpy.nanmax_into" | "ndarray.numpy.nanvar_into" | "ndarray.numpy.nanstd_into" | "ndarray.numpy.nancumsum_flat" | "ndarray.numpy.nancumprod_flat" | "ndarray.numpy.nancumsum_as" | "ndarray.numpy.nancumprod_as" | "ndarray.numpy.nancumsum_into" | "ndarray.numpy.nancumprod_into")
}

impl Worker {
    pub(super) fn execute_chunk_48(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.numpy.nanstd_as" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims), Arg::Int(correction), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanstd_as expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanstd_as(arg_ints(axes), *keep_dims, *correction, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanargmin_axes" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanargmin_axes expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanargmin_axes(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nanargmax_axes" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanargmax_axes expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanargmax_axes(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nansum_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nansum_into expects its declared arguments");
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
                match value_destination.nansum_into(value_array, arg_ints(axes), *keep_dims) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nanprod_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanprod_into expects its declared arguments");
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
                match value_destination.nanprod_into(value_array, arg_ints(axes), *keep_dims) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nanmean_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanmean_into expects its declared arguments");
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
                match value_destination.nanmean_into(value_array, arg_ints(axes), *keep_dims) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nanmin_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanmin_into expects its declared arguments");
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
                match value_destination.nanmin_into(value_array, arg_ints(axes), *keep_dims) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nanmax_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanmax_into expects its declared arguments");
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
                match value_destination.nanmax_into(value_array, arg_ints(axes), *keep_dims) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nanvar_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims), Arg::Int(correction)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanvar_into expects its declared arguments");
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
                match value_destination.nanvar_into(value_array, arg_ints(axes), *keep_dims, *correction) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nanstd_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims), Arg::Int(correction)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanstd_into expects its declared arguments");
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
                match value_destination.nanstd_into(value_array, arg_ints(axes), *keep_dims, *correction) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nancumsum_flat" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nancumsum_flat expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nancumsum_flat() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nancumprod_flat" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nancumprod_flat expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nancumprod_flat() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nancumsum_as" => {
                let [Arg::Handle(array), Arg::Int(axis), Arg::Int(requested_dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nancumsum_as expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nancumsum_as(*axis, *requested_dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nancumprod_as" => {
                let [Arg::Handle(array), Arg::Int(axis), Arg::Int(requested_dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nancumprod_as expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nancumprod_as(*axis, *requested_dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.nancumsum_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nancumsum_into expects its declared arguments");
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
                match value_destination.nancumsum_into(value_array, *axis) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nancumprod_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Int(axis)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nancumprod_into expects its declared arguments");
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
                match value_destination.nancumprod_into(value_array, *axis) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            _ => unreachable!("operation routed to the wrong generated dispatch chunk"),
        }
    }
}
