#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use terlan_ndarray_native::{Array, CAbiError, live_count};

fn main() {
    let mut worker = Worker::default();
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let payload = match line {
            Ok(line) => worker.execute_line(&line),
            Err(error) => protocol_error("native_read_error", &error.to_string()),
        };
        if writeln!(stdout, "{payload}").is_err() || stdout.flush().is_err() {
            break;
        }
    }
}

#[derive(Default)]
struct Worker {
    next_id: u64,
    handles: HashMap<u64, HandleEntry>,
}

struct HandleEntry {
    generation: u64,
    value: HandleValue,
}

enum HandleValue {
    Array(Array),
}

impl HandleValue {
    fn type_name(&self) -> &'static str {
        match self {
            HandleValue::Array(_) => "ndarray.Array.Array",
        }
    }
}

#[derive(Clone)]
struct HandleArg {
    id: u64,
    generation: u64,
    type_name: String,
}

enum Arg {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Bools(Vec<bool>),
    EmptyList,
    Handle(HandleArg),
}

struct Request {
    request_id: u64,
    operation: String,
    args: Vec<Arg>,
}

impl Worker {
    fn execute_line(&mut self, line: &str) -> String {
        let request = match parse_request(line) {
            Ok(request) => request,
            Err(error) => {
                return request_id(line)
                    .map(|request_id| format!("reply {request_id} 1 {error}"))
                    .unwrap_or(error);
            }
        };
        let request_id = request.request_id;
        let payload = self.execute(request);
        format!("reply {request_id} 1 {payload}")
    }

    fn execute(&mut self, request: Request) -> String {
        match request.operation.as_str() {
            "ndarray.array.from_bools" => {
                let [values @ (Arg::Bools(_) | Arg::EmptyList), shape @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_bools expects its declared arguments");
                };
                let value = match Array::from_bools(arg_bools(values), arg_ints(shape)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Array(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.from_ints" => {
                let [values @ (Arg::Ints(_) | Arg::EmptyList), shape @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_ints expects its declared arguments");
                };
                let value = match Array::from_ints(arg_ints(values), arg_ints(shape)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Array(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("ndarray.Array.Array"))
            }
            "ndarray.array.from_floats" => {
                let [values @ (Arg::Floats(_) | Arg::EmptyList), shape @ (Arg::Ints(_) | Arg::EmptyList)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "from_floats expects its declared arguments");
                };
                let value = match Array::from_floats(arg_floats(values), arg_ints(shape)) {
                    Ok(value) => value,
                    Err(error) => return native_error(&error),
                };
                self.next_id += 1;
                let id = self.next_id;
                self.handles.insert(id, HandleEntry { generation: 1, value: HandleValue::Array(value) });
                format!("ok_handle {id} 1 {}", STANDARD.encode("ndarray.Array.Array"))
            }
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
            "ndarray.array.dispose" => {
                let [Arg::Handle(array)] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "dispose expects its declared arguments");
                };
                if let Err(error) = self.validate(array, "ndarray.Array.Array") {
                    return error;
                }
                self.handles.remove(&array.id);
                "ok_unit".to_string()
            }
            "ndarray.array.live_count" => {
                let [] = request.args.as_slice() else {
                    return protocol_error("invalid_arguments", "live_count expects its declared arguments");
                };
                format!("ok_int {}", live_count())
            }

            _ => protocol_error("unknown_operation", &request.operation),
        }
    }

    fn validate(&self, handle: &HandleArg, expected_type: &str) -> Result<(), String> {
        if handle.type_name != expected_type {
            return Err(protocol_error("handle_type_mismatch", &handle.type_name));
        }
        match self.handles.get(&handle.id) {
            Some(entry)
                if entry.generation == handle.generation
                    && entry.value.type_name() == expected_type => Ok(()),
            Some(_) => Err(protocol_error(
                "handle_storage_mismatch",
                "NativeBoundary handle resource type does not match",
            )),
            _ => Err(protocol_error("stale_handle", "NativeBoundary handle is stale")),
        }
    }

    fn live_array(&self, handle: &HandleArg) -> Result<&Array, String> {
        self.validate(handle, "ndarray.Array.Array")?;
        let HandleValue::Array(value) = &self.handles.get(&handle.id).expect("validated handle").value;
        Ok(value)
    }

    #[allow(dead_code)]
    fn live_array_mut(&mut self, handle: &HandleArg) -> Result<&mut Array, String> {
        self.validate(handle, "ndarray.Array.Array")?;
        let HandleValue::Array(value) = &mut self.handles.get_mut(&handle.id).expect("validated handle").value;
        Ok(value)
    }


}

fn request_id(line: &str) -> Option<u64> {
    let mut fields = line.split_whitespace();
    (fields.next() == Some("call"))
        .then(|| fields.next()?.parse::<u64>().ok())
        .flatten()
}

fn parse_request(line: &str) -> Result<Request, String> {
    let mut fields = line.split_whitespace();
    if fields.next() != Some("call") {
        return Err(protocol_error("invalid_request", "expected call request"));
    }
    let request_id = fields
        .next()
        .ok_or_else(|| protocol_error("invalid_request", "missing request id"))?
        .parse::<u64>()
        .map_err(|error| protocol_error("invalid_request", &error.to_string()))?;
    let operation = decode_text(fields.next().ok_or_else(|| {
        protocol_error("invalid_request", "missing encoded operation")
    })?)?;
    let args = fields.map(parse_arg).collect::<Result<Vec<_>, _>>()?;
    for argument in &args {
        validate_decoded_arg_shape(argument);
    }
    Ok(Request { request_id, operation, args })
}

fn parse_arg(value: &str) -> Result<Arg, String> {
    // The VM uses `ls:` for an empty list. Preserve the empty value here and
    // resolve it against each generated operation's declared list type.
    if value == "ls:" {
        return Ok(Arg::EmptyList);
    }
    if let Some(value) = value.strip_prefix("i:") {
        return value
            .parse::<i64>()
            .map(Arg::Int)
            .map_err(|error| protocol_error("invalid_argument", &error.to_string()));
    }
    if let Some(value) = value.strip_prefix("f:") {
        return value
            .parse::<f64>()
            .map(Arg::Float)
            .map_err(|error| protocol_error("invalid_argument", &error.to_string()));
    }
    if let Some(value) = value.strip_prefix("b:") {
        return value
            .parse::<bool>()
            .map(Arg::Bool)
            .map_err(|error| protocol_error("invalid_argument", &error.to_string()));
    }
    if let Some(value) = value.strip_prefix("s:") {
        return decode_text(value).map(Arg::String);
    }
    if let Some(value) = value.strip_prefix("li:") {
        if value.is_empty() {
            return Ok(Arg::Ints(Vec::new()));
        }
        return value
            .split(',')
            .map(|value| {
                value.parse::<i64>().map_err(|error| {
                    protocol_error("invalid_argument", &error.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Arg::Ints);
    }
    if let Some(value) = value.strip_prefix("lf:") {
        if value.is_empty() {
            return Ok(Arg::Floats(Vec::new()));
        }
        return value
            .split(',')
            .map(|value| {
                value.parse::<f64>().map_err(|error| {
                    protocol_error("invalid_argument", &error.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Arg::Floats);
    }
    if let Some(value) = value.strip_prefix("lb:") {
        if value.is_empty() {
            return Ok(Arg::Bools(Vec::new()));
        }
        return value
            .split(',')
            .map(|value| {
                value.parse::<bool>().map_err(|error| {
                    protocol_error("invalid_argument", &error.to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Arg::Bools);
    }
    if let Some(value) = value.strip_prefix("h:") {
        let fields = value.split(':').collect::<Vec<_>>();
        let [id, generation, type_name] = fields.as_slice() else {
            return Err(protocol_error("invalid_argument", "malformed handle"));
        };
        return Ok(Arg::Handle(HandleArg {
            id: id.parse().map_err(|error: std::num::ParseIntError| protocol_error("invalid_argument", &error.to_string()))?,
            generation: generation.parse().map_err(|error: std::num::ParseIntError| protocol_error("invalid_argument", &error.to_string()))?,
            type_name: decode_text(type_name)?,
        }));
    }
    Err(protocol_error("invalid_argument", "unsupported argument encoding"))
}

fn decode_text(value: &str) -> Result<String, String> {
    let bytes = STANDARD
        .decode(value)
        .map_err(|error| protocol_error("invalid_base64", &error.to_string()))?;
    String::from_utf8(bytes)
        .map_err(|error| protocol_error("invalid_utf8", &error.to_string()))
}

fn arg_ints(value: &Arg) -> &[i64] {
    match value {
        Arg::Ints(values) => values.as_slice(),
        Arg::EmptyList => &[],
        _ => unreachable!("generated argument pattern admits only integer lists"),
    }
}

fn arg_floats(value: &Arg) -> &[f64] {
    match value {
        Arg::Floats(values) => values.as_slice(),
        Arg::EmptyList => &[],
        _ => unreachable!("generated argument pattern admits only float lists"),
    }
}

fn arg_bools(value: &Arg) -> &[bool] {
    match value {
        Arg::Bools(values) => values.as_slice(),
        Arg::EmptyList => &[],
        _ => unreachable!("generated argument pattern admits only boolean lists"),
    }
}

fn validate_decoded_arg_shape(value: &Arg) {
    match value {
        Arg::Int(value) => { let _ = value; }
        Arg::Float(value) => { let _ = value; }
        Arg::Bool(value) => { let _ = value; }
        Arg::String(value) => { let _ = value; }
        Arg::Ints(_) => { let _ = arg_ints(value); }
        Arg::Floats(_) => { let _ = arg_floats(value); }
        Arg::Bools(_) => { let _ = arg_bools(value); }
        Arg::EmptyList => {}
        Arg::Handle(value) => { let _ = value; }
    }
}

fn native_error(error: &CAbiError) -> String {
    protocol_error(
        &format!("c_abi_status_{}", error.status),
        error.operation,
    )
}

fn protocol_error(code: &str, message: &str) -> String {
    format!("err {} {}", STANDARD.encode(code), STANDARD.encode(message))
}
