use super::*;

pub(super) fn accepts_chunk_41(operation: &str) -> bool {
    matches!(operation, "ndarray.numpy.nanmedian_axes" | "ndarray.numpy.median_into" | "ndarray.numpy.nanmedian_into" | "ndarray.numpy.quantile_into" | "ndarray.numpy.percentile_into" | "ndarray.numpy.quantiles_into" | "ndarray.numpy.percentiles_into" | "ndarray.numpy.nanquantile_into" | "ndarray.numpy.nanpercentile_into" | "ndarray.numpy.nanquantiles_into" | "ndarray.numpy.nanpercentiles_into" | "ndarray.numpy.packbits" | "ndarray.numpy.unpackbits" | "ndarray.numpy.unpackbits_all" | "ndarray.numpy.packbits_axis" | "ndarray.numpy.unpackbits_axis")
}

impl Worker {
    pub(super) fn execute_chunk_41(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.numpy.nanmedian_axes" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanmedian_axes expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.nanmedian_axes(arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.median_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "median_into expects its declared arguments");
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
                match value_destination.median_into(value_array, arg_ints(axes), *keep_dims) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nanmedian_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanmedian_into expects its declared arguments");
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
                match value_destination.nanmedian_into(value_array, arg_ints(axes), *keep_dims) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.quantile_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Float(quantile), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(interpolation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "quantile_into expects its declared arguments");
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
                match value_destination.quantile_into(value_array, *quantile, arg_ints(axes), *keep_dims, *interpolation) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.percentile_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Float(percentile), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(interpolation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "percentile_into expects its declared arguments");
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
                match value_destination.percentile_into(value_array, *percentile, arg_ints(axes), *keep_dims, *interpolation) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.quantiles_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), quantiles @ (Arg::Floats(_) | Arg::EmptyList), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(interpolation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "quantiles_into expects its declared arguments");
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
                match value_destination.quantiles_into(value_array, arg_floats(quantiles), arg_ints(axes), *keep_dims, *interpolation) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.percentiles_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), percentiles @ (Arg::Floats(_) | Arg::EmptyList), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(interpolation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "percentiles_into expects its declared arguments");
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
                match value_destination.percentiles_into(value_array, arg_floats(percentiles), arg_ints(axes), *keep_dims, *interpolation) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nanquantile_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Float(quantile), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(interpolation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanquantile_into expects its declared arguments");
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
                match value_destination.nanquantile_into(value_array, *quantile, arg_ints(axes), *keep_dims, *interpolation) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nanpercentile_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), Arg::Float(percentile), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(interpolation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanpercentile_into expects its declared arguments");
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
                match value_destination.nanpercentile_into(value_array, *percentile, arg_ints(axes), *keep_dims, *interpolation) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nanquantiles_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), quantiles @ (Arg::Floats(_) | Arg::EmptyList), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(interpolation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanquantiles_into expects its declared arguments");
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
                match value_destination.nanquantiles_into(value_array, arg_floats(quantiles), arg_ints(axes), *keep_dims, *interpolation) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.nanpercentiles_into" => {
                let [Arg::Handle(destination), Arg::Handle(array), percentiles @ (Arg::Floats(_) | Arg::EmptyList), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(keep_dims), Arg::Int(interpolation)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "nanpercentiles_into expects its declared arguments");
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
                match value_destination.nanpercentiles_into(value_array, arg_floats(percentiles), arg_ints(axes), *keep_dims, *interpolation) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.packbits" => {
                let [Arg::Handle(array), Arg::Int(bit_order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "packbits expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.packbits(*bit_order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.unpackbits" => {
                let [Arg::Handle(array), Arg::Int(count), Arg::Int(bit_order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "unpackbits expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.unpackbits(*count, *bit_order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.unpackbits_all" => {
                let [Arg::Handle(array), Arg::Int(bit_order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "unpackbits_all expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.unpackbits_all(*bit_order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.packbits_axis" => {
                let [Arg::Handle(array), Arg::Int(axis), Arg::Int(bit_order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "packbits_axis expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.packbits_axis(*axis, *bit_order) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.unpackbits_axis" => {
                let [Arg::Handle(array), Arg::Int(axis), Arg::Int(count), Arg::Int(bit_order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "unpackbits_axis expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.unpackbits_axis(*axis, *count, *bit_order) {
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
