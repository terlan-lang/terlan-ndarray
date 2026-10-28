use super::*;

pub(super) fn accepts_chunk_46(operation: &str) -> bool {
    matches!(operation, "ndarray.numpy.select" | "ndarray.numpy.shares_memory" | "ndarray.numpy.shares_memory_with_work" | "ndarray.numpy.tril_indices" | "ndarray.numpy.tril_indices_from" | "ndarray.numpy.triu_indices" | "ndarray.numpy.triu_indices_from" | "ndarray.numpy.unravel_index_native" | "ndarray.numpy.vsplit" | "ndarray.numpy.vstack" | "ndarray.numpy.average_weighted" | "ndarray.numpy.average_with_sum" | "ndarray.numpy.average_weighted_with_sum" | "ndarray.numpy.covariance_typed" | "ndarray.numpy.covariance_pair" | "ndarray.numpy.covariance_frequency_weighted")
}

impl Worker {
    pub(super) fn execute_chunk_46(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.numpy.select" => {
                let [conditions @ (Arg::Handles(_) | Arg::EmptyList), choices @ (Arg::Handles(_) | Arg::EmptyList), Arg::Handle(default_value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "select expects its declared arguments");
                };
                let value_conditions = {
                    let mut values = Vec::new();
                    for handle in arg_handles(conditions) {
                        let value = match self.live_array(handle) {
                            Ok(value) => value,
                            Err(error) => return error,
                        };
                        values.push(value);
                    }
                    values
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
                let value_default_value = match self.live_array(default_value) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match select(value_conditions.as_slice(), value_choices.as_slice(), value_default_value) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.shares_memory" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "shares_memory expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_lhs.shares_memory(value_rhs) { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.shares_memory_with_work" => {
                let [Arg::Handle(lhs), Arg::Handle(rhs), Arg::Int(max_work)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "shares_memory_with_work expects its declared arguments");
                };
                let value_lhs = match self.live_array(lhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_lhs.shares_memory_with_work(value_rhs, *max_work) { Ok(value) => format!("ok_bool {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.tril_indices" => {
                let [Arg::Int(rows), Arg::Int(offset), Arg::Int(columns)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "tril_indices expects its declared arguments");
                };
                let value = match Array::tril_indices(*rows, *offset, *columns) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.tril_indices_from" => {
                let [Arg::Handle(array), Arg::Int(offset)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "tril_indices_from expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.tril_indices_from(*offset) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.triu_indices" => {
                let [Arg::Int(rows), Arg::Int(offset), Arg::Int(columns)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "triu_indices expects its declared arguments");
                };
                let value = match Array::triu_indices(*rows, *offset, *columns) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.triu_indices_from" => {
                let [Arg::Handle(array), Arg::Int(offset)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "triu_indices_from expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.triu_indices_from(*offset) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.unravel_index_native" => {
                let [Arg::Handle(indexes), dimensions @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(order)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "unravel_index_native expects its declared arguments");
                };
                let value_indexes = match self.live_array(indexes) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_indexes.unravel_index_native(arg_ints(dimensions), *order) {
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
            "ndarray.numpy.vsplit" => {
                let [Arg::Handle(array), Arg::Int(sections)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "vsplit expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.vsplit(*sections) {
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
            "ndarray.numpy.vstack" => {
                let [arrays @ (Arg::Handles(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "vstack expects its declared arguments");
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
                let value = match vstack(value_arrays.as_slice()) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.average_weighted" => {
                let [Arg::Handle(array), Arg::Handle(weights), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "average_weighted expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weights = match self.live_array(weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.average_weighted(value_weights, arg_ints(axes), *keep_dims) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.average_with_sum" => {
                let [Arg::Handle(array), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "average_with_sum expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.average_with_sum(arg_ints(axes), *keep_dims) {
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
            "ndarray.numpy.average_weighted_with_sum" => {
                let [Arg::Handle(array), Arg::Handle(weights), axes @ (Arg::Ints(_) | Arg::EmptyList), Arg::Bool(keep_dims)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "average_weighted_with_sum expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weights = match self.live_array(weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.average_weighted_with_sum(value_weights, arg_ints(axes), *keep_dims) {
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
            "ndarray.numpy.covariance_typed" => {
                let [Arg::Handle(array), Arg::Int(row_variables), Arg::Int(correction), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "covariance_typed expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.covariance_typed(*row_variables, *correction, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.covariance_pair" => {
                let [Arg::Handle(array), Arg::Handle(other), Arg::Int(row_variables), Arg::Int(correction), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "covariance_pair expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_other = match self.live_array(other) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.covariance_pair(value_other, *row_variables, *correction, *dtype) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.covariance_frequency_weighted" => {
                let [Arg::Handle(array), Arg::Handle(frequency_weights), Arg::Int(row_variables), Arg::Int(correction), Arg::Int(dtype)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "covariance_frequency_weighted expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_frequency_weights = match self.live_array(frequency_weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.covariance_frequency_weighted(value_frequency_weights, *row_variables, *correction, *dtype) {
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
