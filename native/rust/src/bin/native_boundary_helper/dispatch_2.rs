use super::*;

pub(super) fn accepts_chunk_2(operation: &str) -> bool {
    matches!(operation, "ndarray.array.rank" | "ndarray.array.shape" | "ndarray.array.numel" | "ndarray.array.strides" | "ndarray.array.byte_count" | "ndarray.array.dtype_code" | "ndarray.array.contiguous_code" | "ndarray.array.fortran_contiguous_code" | "ndarray.array.device_type_code" | "ndarray.array.to_bools" | "ndarray.array.to_ints" | "ndarray.array.to_floats" | "ndarray.array.to_ints_cast" | "ndarray.array.to_floats_cast" | "ndarray.array.complex_reals" | "ndarray.array.complex_imags")
}

impl Worker {
    pub(super) fn execute_chunk_2(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.rank" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "rank expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.rank() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.shape" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "shape expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.shape() { Ok(values) if values.is_empty() => "ok_ints".to_string(), Ok(values) => format!("ok_ints {}", values.iter().map(i64::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "ndarray.array.numel" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "numel expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.numel() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.strides" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "strides expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.strides() { Ok(values) if values.is_empty() => "ok_ints".to_string(), Ok(values) => format!("ok_ints {}", values.iter().map(i64::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "ndarray.array.byte_count" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "byte_count expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.byte_count() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.dtype_code" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dtype_code expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.dtype_code() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.contiguous_code" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "contiguous_code expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.contiguous_code() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.fortran_contiguous_code" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "fortran_contiguous_code expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.fortran_contiguous_code() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.device_type_code" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "device_type_code expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.device_type_code() { Ok(value) => format!("ok_int {value}"), Err(error) => native_error(&error) }
            }
            "ndarray.array.to_bools" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "to_bools expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.to_bools() { Ok(values) if values.is_empty() => "ok_bools".to_string(), Ok(values) => format!("ok_bools {}", values.iter().map(bool::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "ndarray.array.to_ints" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "to_ints expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.to_ints() { Ok(values) if values.is_empty() => "ok_ints".to_string(), Ok(values) => format!("ok_ints {}", values.iter().map(i64::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "ndarray.array.to_floats" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "to_floats expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.to_floats() { Ok(values) if values.is_empty() => "ok_floats".to_string(), Ok(values) => format!("ok_floats {}", values.iter().map(f64::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "ndarray.array.to_ints_cast" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "to_ints_cast expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.to_ints_cast() { Ok(values) if values.is_empty() => "ok_ints".to_string(), Ok(values) => format!("ok_ints {}", values.iter().map(i64::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "ndarray.array.to_floats_cast" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "to_floats_cast expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.to_floats_cast() { Ok(values) if values.is_empty() => "ok_floats".to_string(), Ok(values) => format!("ok_floats {}", values.iter().map(f64::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "ndarray.array.complex_reals" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "complex_reals expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.complex_reals() { Ok(values) if values.is_empty() => "ok_floats".to_string(), Ok(values) => format!("ok_floats {}", values.iter().map(f64::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            "ndarray.array.complex_imags" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "complex_imags expects its declared arguments");
                };
                let value_array = match self.live_array(array) {
                    Ok(value) => value,
                    Err(error) => return error,
                };
                match value_array.complex_imags() { Ok(values) if values.is_empty() => "ok_floats".to_string(), Ok(values) => format!("ok_floats {}", values.iter().map(f64::to_string).collect::<Vec<_>>().join(",")), Err(error) => native_error(&error) }
            }
            _ => unreachable!("operation routed to the wrong generated dispatch chunk"),
        }
    }
}
