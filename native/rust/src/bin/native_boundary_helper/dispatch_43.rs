use super::*;

pub(super) fn accepts_chunk_43(operation: &str) -> bool {
    matches!(operation, "ndarray.numpy.diag_indices" | "ndarray.numpy.diag_indices_from" | "ndarray.numpy.dsplit" | "ndarray.numpy.dstack" | "ndarray.numpy.ediff1d" | "ndarray.numpy.fill_diagonal_native" | "ndarray.numpy.full" | "ndarray.numpy.full_like" | "ndarray.numpy.histogram" | "ndarray.numpy.histogram_range" | "ndarray.numpy.histogram_edges" | "ndarray.numpy.histogram_weighted" | "ndarray.numpy.histogram_weighted_range" | "ndarray.numpy.histogram_weighted_edges" | "ndarray.numpy.histogram_estimator" | "ndarray.numpy.histogram_estimator_range")
}

impl Worker {
    pub(super) fn execute_chunk_43(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.numpy.diag_indices" => {
                let [Arg::Int(size), Arg::Int(rank)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "diag_indices expects its declared arguments");
                };
                let value = match Array::diag_indices(*size, *rank) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.diag_indices_from" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "diag_indices_from expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.diag_indices_from() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.dsplit" => {
                let [Arg::Handle(array), Arg::Int(sections)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dsplit expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.dsplit(*sections) {
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
            "ndarray.numpy.dstack" => {
                let [arrays @ (Arg::Handles(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dstack expects its declared arguments");
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
                let value = match dstack(value_arrays.as_slice()) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.ediff1d" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "ediff1d expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.ediff1d() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.fill_diagonal_native" => {
                let [Arg::Handle(array), Arg::Float(value), Arg::Bool(wrap)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "fill_diagonal_native expects its declared arguments");
                };
                let value_array = match self.live_array_mut(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.fill_diagonal_native(*value, *wrap) { Ok(()) => "ok_unit".to_string(), Err(error) => native_error(&error) }
            }
            "ndarray.numpy.full" => {
                let [shape @ (Arg::Ints(_) | Arg::EmptyList), Arg::Float(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "full expects its declared arguments");
                };
                let value = match Array::full(arg_ints(shape), *value) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.full_like" => {
                let [Arg::Handle(array), Arg::Float(value)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "full_like expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.full_like(*value) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.histogram" => {
                let [Arg::Handle(array), Arg::Int(bins), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.numpy_histogram(*bins, *density) {
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
            "ndarray.numpy.histogram_range" => {
                let [Arg::Handle(array), Arg::Int(bins), Arg::Float(lower), Arg::Float(upper), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram_range expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.numpy_histogram_range(*bins, *lower, *upper, *density) {
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
            "ndarray.numpy.histogram_edges" => {
                let [Arg::Handle(array), Arg::Handle(edges), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram_edges expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_edges = match self.live_array(edges) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.numpy_histogram_edges(value_edges, *density) {
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
            "ndarray.numpy.histogram_weighted" => {
                let [Arg::Handle(array), Arg::Handle(weights), Arg::Int(bins), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram_weighted expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weights = match self.live_array(weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.numpy_histogram_weighted(value_weights, *bins, *density) {
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
            "ndarray.numpy.histogram_weighted_range" => {
                let [Arg::Handle(array), Arg::Handle(weights), Arg::Int(bins), Arg::Float(lower), Arg::Float(upper), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram_weighted_range expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weights = match self.live_array(weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.numpy_histogram_weighted_range(value_weights, *bins, *lower, *upper, *density) {
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
            "ndarray.numpy.histogram_weighted_edges" => {
                let [Arg::Handle(array), Arg::Handle(weights), Arg::Handle(edges), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram_weighted_edges expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weights = match self.live_array(weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_edges = match self.live_array(edges) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.numpy_histogram_weighted_edges(value_weights, value_edges, *density) {
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
            "ndarray.numpy.histogram_estimator" => {
                let [Arg::Handle(array), Arg::Int(estimator), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram_estimator expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.numpy_histogram_estimator(*estimator, *density) {
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
            "ndarray.numpy.histogram_estimator_range" => {
                let [Arg::Handle(array), Arg::Int(estimator), Arg::Float(lower), Arg::Float(upper), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram_estimator_range expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.numpy_histogram_estimator_range(*estimator, *lower, *upper, *density) {
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
            _ => unreachable!("operation routed to the wrong generated dispatch chunk"),
        }
    }
}
