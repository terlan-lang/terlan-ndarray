use super::*;

pub(super) fn accepts_chunk_25(operation: &str) -> bool {
    matches!(operation, "ndarray.shape.pad_maximum" | "ndarray.shape.pad_mean" | "ndarray.shape.pad_median" | "ndarray.shape.pad_constant_values" | "ndarray.shape.pad_linear_ramp_values" | "ndarray.shape.pad_statistical" | "ndarray.array.det" | "ndarray.array.least_squares" | "ndarray.array.matrix_rank_general" | "ndarray.array.lstsq" | "ndarray.array.cholesky" | "ndarray.array.qr" | "ndarray.array.svd" | "ndarray.array.singular_values" | "ndarray.array.qr_mode" | "ndarray.array.symmetric_eigen")
}

impl Worker {
    pub(super) fn execute_chunk_25(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.shape.pad_maximum" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList), leading_lengths @ (Arg::Ints(_) | Arg::EmptyList), trailing_lengths @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_maximum expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_maximum(value_array, arg_ints(leading), arg_ints(trailing), arg_ints(leading_lengths), arg_ints(trailing_lengths)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_mean" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList), leading_lengths @ (Arg::Ints(_) | Arg::EmptyList), trailing_lengths @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_mean expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_mean(value_array, arg_ints(leading), arg_ints(trailing), arg_ints(leading_lengths), arg_ints(trailing_lengths)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_median" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList), leading_lengths @ (Arg::Ints(_) | Arg::EmptyList), trailing_lengths @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_median expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_median(value_array, arg_ints(leading), arg_ints(trailing), arg_ints(leading_lengths), arg_ints(trailing_lengths)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_constant_values" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList), Arg::Handle(leading_values), Arg::Handle(trailing_values)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_constant_values expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_leading_values = match self.live_array(leading_values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_trailing_values = match self.live_array(trailing_values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_constant_values(value_array, arg_ints(leading), arg_ints(trailing), value_leading_values, value_trailing_values) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_linear_ramp_values" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList), Arg::Handle(leading_values), Arg::Handle(trailing_values)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_linear_ramp_values expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_leading_values = match self.live_array(leading_values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_trailing_values = match self.live_array(trailing_values) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_linear_ramp_values(value_array, arg_ints(leading), arg_ints(trailing), value_leading_values, value_trailing_values) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.shape.pad_statistical" => {
                let [Arg::Handle(array), leading @ (Arg::Ints(_) | Arg::EmptyList), trailing @ (Arg::Ints(_) | Arg::EmptyList), leading_lengths @ (Arg::Ints(_) | Arg::EmptyList), trailing_lengths @ (Arg::Ints(_) | Arg::EmptyList), Arg::Int(mode)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "pad_statistical expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match pad_statistical(value_array, arg_ints(leading), arg_ints(trailing), arg_ints(leading_lengths), arg_ints(trailing_lengths), *mode) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.det" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "det expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.det() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.least_squares" => {
                let [Arg::Handle(matrix), Arg::Handle(rhs), Arg::Float(tolerance)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "least_squares expects its declared arguments");
                };
                let value_matrix = match self.live_array(matrix) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_matrix.least_squares(value_rhs, *tolerance) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.matrix_rank_general" => {
                let [Arg::Handle(array), Arg::Float(tolerance), Arg::Bool(hermitian)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "matrix_rank_general expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.matrix_rank_general(*tolerance, *hermitian) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.lstsq" => {
                let [Arg::Handle(coefficients), Arg::Handle(rhs), Arg::Float(rcond)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "lstsq_general expects its declared arguments");
                };
                let value_coefficients = match self.live_array(coefficients) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value_rhs = match self.live_array(rhs) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_coefficients.lstsq_general(value_rhs, *rcond) {
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
            "ndarray.array.cholesky" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "cholesky expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.cholesky() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.qr" => {
                let [Arg::Handle(array), Arg::Bool(reduced)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "qr_factors expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.qr_factors(*reduced) {
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
            "ndarray.array.svd" => {
                let [Arg::Handle(array), Arg::Bool(full_matrices)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "svd expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.svd(*full_matrices) {
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
            "ndarray.array.singular_values" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "singular_values expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let value = match value_array.singular_values() {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                let (id, generation) = match self.store_handle(HandleValue::Array(value)) {
                    Ok(handle) => handle,
                    Err(error) => return error,
                };
                format!("ok_handle {} {id} {generation} {}", STANDARD.encode(self.owner.as_bytes()), STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.qr_mode" => {
                let [Arg::Handle(array), Arg::Int(mode)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "qr_mode expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.qr_mode(*mode) {
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
            "ndarray.array.symmetric_eigen" => {
                let [Arg::Handle(array), Arg::Float(tolerance)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "symmetric_eigen expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                let values = match value_array.symmetric_eigen(*tolerance) {
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
