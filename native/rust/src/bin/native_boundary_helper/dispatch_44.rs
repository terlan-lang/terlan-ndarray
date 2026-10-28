use super::*;

pub(super) fn accepts_chunk_44(operation: &str) -> bool {
    matches!(operation, "ndarray.numpy.histogram_bin_edges" | "ndarray.numpy.histogram_bin_edges_estimator" | "ndarray.numpy.histogram2d" | "ndarray.numpy.histogram2d_range" | "ndarray.numpy.histogram2d_edges" | "ndarray.numpy.histogram2d_x_edges" | "ndarray.numpy.histogram2d_y_edges" | "ndarray.numpy.histogramdd" | "ndarray.numpy.histogramdd_bins_array" | "ndarray.numpy.histogramdd_range_arrays" | "ndarray.numpy.histogramdd_edges_packed" | "ndarray.numpy.histogramdd_weighted" | "ndarray.numpy.histogramdd_weighted_bins_array" | "ndarray.numpy.histogramdd_weighted_range_arrays" | "ndarray.numpy.histogramdd_weighted_edges_packed" | "ndarray.numpy.hsplit")
}

impl Worker {
    pub(super) fn execute_chunk_44(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.numpy.histogram_bin_edges" => {
                let [Arg::Handle(array), Arg::Int(bins)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram_bin_edges expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.numpy_histogram_bin_edges(*bins) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.histogram_bin_edges_estimator" => {
                let [Arg::Handle(array), Arg::Int(estimator)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram_bin_edges_estimator expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.numpy_histogram_bin_edges_estimator(*estimator) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.numpy.histogram2d" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Int(bins), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram2d expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_x.histogram2d(value_y, *bins, *density) {
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
            "ndarray.numpy.histogram2d_range" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Int(x_bins), Arg::Int(y_bins), Arg::Handle(minima), Arg::Handle(maxima), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram2d_range expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_minima = match self.live_array(minima) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_maxima = match self.live_array(maxima) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_x.histogram2d_range(value_y, *x_bins, *y_bins, value_minima, value_maxima, *density) {
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
            "ndarray.numpy.histogram2d_edges" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(x_edges), Arg::Handle(y_edges), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram2d_edges expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_x_edges = match self.live_array(x_edges) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y_edges = match self.live_array(y_edges) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_x.histogram2d_edges(value_y, value_x_edges, value_y_edges, *density) {
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
            "ndarray.numpy.histogram2d_x_edges" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Handle(x_edges), Arg::Int(y_bins), Arg::Float(y_lower), Arg::Float(y_upper), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram2d_x_edges expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_x_edges = match self.live_array(x_edges) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_x.histogram2d_x_edges(value_y, value_x_edges, *y_bins, *y_lower, *y_upper, *density) {
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
            "ndarray.numpy.histogram2d_y_edges" => {
                let [Arg::Handle(x), Arg::Handle(y), Arg::Int(x_bins), Arg::Float(x_lower), Arg::Float(x_upper), Arg::Handle(y_edges), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogram2d_y_edges expects its declared arguments");
                };
                let value_x = match self.live_array(x) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y = match self.live_array(y) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_y_edges = match self.live_array(y_edges) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_x.histogram2d_y_edges(value_y, *x_bins, *x_lower, *x_upper, value_y_edges, *density) {
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
            "ndarray.numpy.histogramdd" => {
                let [Arg::Handle(samples), Arg::Int(bins), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogramdd expects its declared arguments");
                };
                let value_samples = match self.live_array(samples) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_samples.numpy_histogramdd(*bins, *density) {
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
            "ndarray.numpy.histogramdd_bins_array" => {
                let [Arg::Handle(samples), Arg::Handle(bins), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogramdd_bins_array expects its declared arguments");
                };
                let value_samples = match self.live_array(samples) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_bins = match self.live_array(bins) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_samples.numpy_histogramdd_bins_array(value_bins, *density) {
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
            "ndarray.numpy.histogramdd_range_arrays" => {
                let [Arg::Handle(samples), Arg::Handle(bins), Arg::Handle(minima), Arg::Handle(maxima), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogramdd_range_arrays expects its declared arguments");
                };
                let value_samples = match self.live_array(samples) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_bins = match self.live_array(bins) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_minima = match self.live_array(minima) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_maxima = match self.live_array(maxima) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_samples.numpy_histogramdd_range_arrays(value_bins, value_minima, value_maxima, *density) {
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
            "ndarray.numpy.histogramdd_edges_packed" => {
                let [Arg::Handle(samples), Arg::Handle(edges), Arg::Handle(offsets), Arg::Handle(dtypes), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogramdd_edges_packed expects its declared arguments");
                };
                let value_samples = match self.live_array(samples) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_edges = match self.live_array(edges) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_offsets = match self.live_array(offsets) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_dtypes = match self.live_array(dtypes) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_samples.numpy_histogramdd_edges_packed(value_edges, value_offsets, value_dtypes, *density) {
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
            "ndarray.numpy.histogramdd_weighted" => {
                let [Arg::Handle(samples), Arg::Handle(weights), Arg::Int(bins), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogramdd_weighted expects its declared arguments");
                };
                let value_samples = match self.live_array(samples) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weights = match self.live_array(weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_samples.numpy_histogramdd_weighted(value_weights, *bins, *density) {
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
            "ndarray.numpy.histogramdd_weighted_bins_array" => {
                let [Arg::Handle(samples), Arg::Handle(weights), Arg::Handle(bins), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogramdd_weighted_bins_array expects its declared arguments");
                };
                let value_samples = match self.live_array(samples) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weights = match self.live_array(weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_bins = match self.live_array(bins) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_samples.numpy_histogramdd_weighted_bins_array(value_weights, value_bins, *density) {
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
            "ndarray.numpy.histogramdd_weighted_range_arrays" => {
                let [Arg::Handle(samples), Arg::Handle(weights), Arg::Handle(bins), Arg::Handle(minima), Arg::Handle(maxima), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogramdd_weighted_range_arrays expects its declared arguments");
                };
                let value_samples = match self.live_array(samples) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_weights = match self.live_array(weights) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_bins = match self.live_array(bins) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_minima = match self.live_array(minima) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_maxima = match self.live_array(maxima) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_samples.numpy_histogramdd_weighted_range_arrays(value_weights, value_bins, value_minima, value_maxima, *density) {
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
            "ndarray.numpy.histogramdd_weighted_edges_packed" => {
                let [Arg::Handle(samples), Arg::Handle(weights), Arg::Handle(edges), Arg::Handle(offsets), Arg::Handle(dtypes), Arg::Bool(density)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "histogramdd_weighted_edges_packed expects its declared arguments");
                };
                let value_samples = match self.live_array(samples) {
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
                let value_offsets = match self.live_array(offsets) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_dtypes = match self.live_array(dtypes) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_samples.numpy_histogramdd_weighted_edges_packed(value_weights, value_edges, value_offsets, value_dtypes, *density) {
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
            "ndarray.numpy.hsplit" => {
                let [Arg::Handle(array), Arg::Int(sections)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "hsplit expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.hsplit(*sections) {
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
