//! Native-boundary helper generated from reviewed C ABI metadata.

#![forbid(unsafe_code)]

#[path = "native_boundary_helper/dispatch_0.rs"]
mod native_boundary_dispatch_0;
#[path = "native_boundary_helper/dispatch_1.rs"]
mod native_boundary_dispatch_1;
#[path = "native_boundary_helper/dispatch_2.rs"]
mod native_boundary_dispatch_2;
#[path = "native_boundary_helper/dispatch_3.rs"]
mod native_boundary_dispatch_3;
#[path = "native_boundary_helper/dispatch_4.rs"]
mod native_boundary_dispatch_4;
#[path = "native_boundary_helper/dispatch_5.rs"]
mod native_boundary_dispatch_5;
#[path = "native_boundary_helper/dispatch_6.rs"]
mod native_boundary_dispatch_6;
#[path = "native_boundary_helper/dispatch_7.rs"]
mod native_boundary_dispatch_7;
#[path = "native_boundary_helper/dispatch_8.rs"]
mod native_boundary_dispatch_8;
#[path = "native_boundary_helper/dispatch_9.rs"]
mod native_boundary_dispatch_9;
#[path = "native_boundary_helper/dispatch_10.rs"]
mod native_boundary_dispatch_10;
#[path = "native_boundary_helper/dispatch_11.rs"]
mod native_boundary_dispatch_11;
#[path = "native_boundary_helper/dispatch_12.rs"]
mod native_boundary_dispatch_12;
#[path = "native_boundary_helper/dispatch_13.rs"]
mod native_boundary_dispatch_13;
#[path = "native_boundary_helper/dispatch_14.rs"]
mod native_boundary_dispatch_14;
#[path = "native_boundary_helper/dispatch_15.rs"]
mod native_boundary_dispatch_15;
#[path = "native_boundary_helper/dispatch_16.rs"]
mod native_boundary_dispatch_16;
#[path = "native_boundary_helper/dispatch_17.rs"]
mod native_boundary_dispatch_17;
#[path = "native_boundary_helper/dispatch_18.rs"]
mod native_boundary_dispatch_18;
#[path = "native_boundary_helper/dispatch_19.rs"]
mod native_boundary_dispatch_19;
#[path = "native_boundary_helper/dispatch_20.rs"]
mod native_boundary_dispatch_20;
#[path = "native_boundary_helper/dispatch_21.rs"]
mod native_boundary_dispatch_21;
#[path = "native_boundary_helper/dispatch_22.rs"]
mod native_boundary_dispatch_22;
#[path = "native_boundary_helper/dispatch_23.rs"]
mod native_boundary_dispatch_23;
#[path = "native_boundary_helper/dispatch_24.rs"]
mod native_boundary_dispatch_24;
#[path = "native_boundary_helper/dispatch_25.rs"]
mod native_boundary_dispatch_25;
#[path = "native_boundary_helper/dispatch_26.rs"]
mod native_boundary_dispatch_26;
#[path = "native_boundary_helper/dispatch_27.rs"]
mod native_boundary_dispatch_27;
#[path = "native_boundary_helper/dispatch_28.rs"]
mod native_boundary_dispatch_28;
#[path = "native_boundary_helper/dispatch_29.rs"]
mod native_boundary_dispatch_29;
#[path = "native_boundary_helper/dispatch_30.rs"]
mod native_boundary_dispatch_30;
#[path = "native_boundary_helper/dispatch_31.rs"]
mod native_boundary_dispatch_31;
#[path = "native_boundary_helper/dispatch_32.rs"]
mod native_boundary_dispatch_32;
#[path = "native_boundary_helper/dispatch_33.rs"]
mod native_boundary_dispatch_33;
#[path = "native_boundary_helper/dispatch_34.rs"]
mod native_boundary_dispatch_34;
#[path = "native_boundary_helper/dispatch_35.rs"]
mod native_boundary_dispatch_35;
#[path = "native_boundary_helper/dispatch_36.rs"]
mod native_boundary_dispatch_36;
#[path = "native_boundary_helper/dispatch_37.rs"]
mod native_boundary_dispatch_37;
#[path = "native_boundary_helper/dispatch_38.rs"]
mod native_boundary_dispatch_38;
#[path = "native_boundary_helper/dispatch_39.rs"]
mod native_boundary_dispatch_39;
#[path = "native_boundary_helper/dispatch_40.rs"]
mod native_boundary_dispatch_40;
#[path = "native_boundary_helper/dispatch_41.rs"]
mod native_boundary_dispatch_41;
#[path = "native_boundary_helper/dispatch_42.rs"]
mod native_boundary_dispatch_42;
#[path = "native_boundary_helper/dispatch_43.rs"]
mod native_boundary_dispatch_43;
#[path = "native_boundary_helper/dispatch_44.rs"]
mod native_boundary_dispatch_44;
#[path = "native_boundary_helper/dispatch_45.rs"]
mod native_boundary_dispatch_45;
#[path = "native_boundary_helper/dispatch_46.rs"]
mod native_boundary_dispatch_46;
#[path = "native_boundary_helper/dispatch_47.rs"]
mod native_boundary_dispatch_47;
#[path = "native_boundary_helper/dispatch_48.rs"]
mod native_boundary_dispatch_48;
#[path = "native_boundary_helper/dispatch_49.rs"]
mod native_boundary_dispatch_49;
#[path = "native_boundary_helper/dispatch_50.rs"]
mod native_boundary_dispatch_50;

use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use terlan_ndarray_native::{Array, ArrowArray, CAbiError, RandomStream, atleast_1d, atleast_2d, atleast_3d, bartlett, basic_slice, blackman, block_native, cheb2poly, chebadd, chebcompanion, chebder, chebder_axis, chebfit, chebfit_degrees, chebfit_weighted_degrees, chebfromroots, chebgauss, chebgrid2d, chebgrid3d, chebint, chebint_axis, chebinterpolate, chebline, chebmul, chebmulx, chebpow, chebpts1, chebpts2, chebroots, chebsub, chebtrim, chebval, chebval2d, chebval3d, chebval_tensor, chebvander, chebvander2d, chebvander3d, chebweight, column_stack, concatenate, concatenate_flat, concatenate_flat_into, concatenate_flat_with_dtype, concatenate_into, concatenate_with_dtype, diagflat, dstack, einsum, einsum_ordered, empty_like, eye_ordered, flatten, flatten_ordered, flip_all, fliplr, flipud, full_like_float, full_like_int, geomspace, geomspace_arrays, hamming, hanning, hstack, indices, indices_sparse, kaiser, leg2poly, legadd, legcompanion, legder, legder_axis, legfit, legfit_degrees, legfit_weighted_degrees, legfromroots, leggauss, leggrid2d, leggrid3d, legint, legint_axis, legline, legmul, legmulx, legpow, legroots, legsub, legtrim, legval, legval2d, legval3d, legval_tensor, legvander, legvander2d, legvander3d, legweight, linspace_arrays, linspace_steps, live_count, logspace, logspace_arrays, moveaxis, multi_dot, ones_like, pad_constant, pad_constant_values, pad_edge, pad_empty, pad_linear_ramp, pad_linear_ramp_values, pad_maximum, pad_mean, pad_median, pad_minimum, pad_reflect, pad_reflect_odd, pad_statistical, pad_symmetric, pad_symmetric_odd, pad_wrap, poly2cheb, poly2leg, polygrid2d, polygrid3d, polyline, polyval2d, polyval3d, polyvander2d, polyvander3d, ravel, ravel_ordered, rollaxis, rot90, row_stack, select, stack, take_indices, take_indices_mode, tri, tril, triu, vander, vstack, zeros_like};

const MAX_ADAPTER_FRAME_BYTES: usize = 1048576;
const MAX_ADAPTER_TRANSFER_BYTES: usize = 16777216;

fn main() {
    let mut worker = match Worker::new() {
        Ok(worker) => worker,
        Err(error) => {
            eprintln!("native worker identity initialization failed: {error}");
            return;
        }
    };
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut stdout = io::stdout().lock();
    loop {
        let mut frame = Vec::new();
        let read = Read::by_ref(&mut input)
            .take((MAX_ADAPTER_FRAME_BYTES + 1) as u64)
            .read_until(b'\n', &mut frame);
        let (payload, terminate) = match read {
            Ok(0) => break,
            Ok(_) if frame.len() > MAX_ADAPTER_FRAME_BYTES => (
                protocol_error("frame_too_large", "native adapter frame exceeds its bound"),
                true,
            ),
            Ok(_) => match String::from_utf8(frame) {
                Ok(line) => (
                    worker.execute_line(line.trim_end_matches(['\r', '\n'])),
                    false,
                ),
                Err(error) => (protocol_error("invalid_utf8", &error.to_string()), false),
            },
            Err(error) => (protocol_error("native_read_error", &error.to_string()), true),
        };
        if writeln!(stdout, "{payload}").is_err() || stdout.flush().is_err() || terminate {
            break;
        }
    }
}

struct Worker {
    owner: String,
    last_request_id: Option<u64>,
    next_id: u64,
    free_ids: Vec<u64>,
    generations: HashMap<u64, u64>,
    handles: HashMap<u64, HandleEntry>,
    request_handles: HashMap<u64, HandleArg>,
}

struct HandleEntry {
    generation: u64,
    value: HandleValue,
}

enum HandleValue {
    Array(Array),
    ArrowArray(ArrowArray),
    RandomStream(RandomStream),
}

impl HandleValue {
    fn type_name(&self) -> &'static str {
        match self {
            HandleValue::Array(_) => "ndarray.Array.Array",
            HandleValue::ArrowArray(_) => "ndarray.ArrowArray.ArrowArray",
            HandleValue::RandomStream(_) => "ndarray.RandomStream.RandomStream",
        }
    }
}

#[derive(Clone)]
struct HandleArg {
    owner: String,
    id: u64,
    generation: u64,
    type_name: String,
}

enum Arg {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Bytes(Vec<u8>),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Bools(Vec<bool>),
    Handles(Vec<HandleArg>),
    EmptyList,
    Handle(HandleArg),
}

struct Request {
    operation: String,
    args: Vec<Arg>,
}

impl Worker {
    fn new() -> Result<Self, getrandom::Error> {
        let mut owner = [0_u8; 32];
        getrandom::fill(&mut owner)?;
        Ok(Self {
            owner: STANDARD.encode(owner),
            last_request_id: None,
            next_id: 0,
            free_ids: Vec::new(),
            generations: HashMap::new(),
            handles: HashMap::new(),
            request_handles: HashMap::new(),
        })
    }

    fn store_handle(&mut self, value: HandleValue) -> Result<(u64, u64), String> {
        while let Some(id) = self.free_ids.pop() {
            let previous = self.generations.get(&id).copied().unwrap_or_default();
            if let Some(generation) = previous.checked_add(1) {
                self.generations.insert(id, generation);
                self.handles.insert(id, HandleEntry { generation, value });
                return Ok((id, generation));
            }
        }
        let Some(id) = self.next_id.checked_add(1) else {
            return Err(protocol_error(
                "resource_table_exhausted",
                "NativeBoundary handle table exhausted",
            ));
        };
        self.next_id = id;
        let generation = 1;
        self.generations.insert(id, generation);
        self.handles.insert(id, HandleEntry { generation, value });
        Ok((id, generation))
    }

    fn release_handle(&mut self, id: u64) {
        if self.handles.remove(&id).is_some() {
            self.free_ids.push(id);
        }
    }

    fn execute_line(&mut self, line: &str) -> String {
        let Some(request_id) = request_id(line) else {
            return match parse_request(line) {
                Ok(_) => protocol_error("invalid_request", "request id is missing"),
                Err(error) => error,
            };
        };
        if self
            .last_request_id
            .is_some_and(|last_request_id| request_id <= last_request_id)
        {
            return format!(
                "reply {request_id} 1 {}",
                protocol_error(
                    "request_not_monotonic",
                    "native adapter request id was already completed"
                )
            );
        }
        self.last_request_id = Some(request_id);
        let resolved = match resolve_request_references(line, &self.request_handles) {
            Ok(resolved) => resolved,
            Err(error) => return format!("reply {request_id} 1 {error}"),
        };
        let request = match parse_request(&resolved) {
            Ok(request) => request,
            Err(error) => return format!("reply {request_id} 1 {error}"),
        };
        let payload = self.execute(request);
        if let Some(handle) = response_handle(&payload) {
            self.request_handles.insert(request_id, handle);
        }
        format!("reply {request_id} 1 {payload}")
    }

    fn execute(&mut self, request: Request) -> String {
        if native_boundary_dispatch_0::accepts_chunk_0(request.operation.as_str()) {
            return self.execute_chunk_0(request);
        }
        if native_boundary_dispatch_1::accepts_chunk_1(request.operation.as_str()) {
            return self.execute_chunk_1(request);
        }
        if native_boundary_dispatch_2::accepts_chunk_2(request.operation.as_str()) {
            return self.execute_chunk_2(request);
        }
        if native_boundary_dispatch_3::accepts_chunk_3(request.operation.as_str()) {
            return self.execute_chunk_3(request);
        }
        if native_boundary_dispatch_4::accepts_chunk_4(request.operation.as_str()) {
            return self.execute_chunk_4(request);
        }
        if native_boundary_dispatch_5::accepts_chunk_5(request.operation.as_str()) {
            return self.execute_chunk_5(request);
        }
        if native_boundary_dispatch_6::accepts_chunk_6(request.operation.as_str()) {
            return self.execute_chunk_6(request);
        }
        if native_boundary_dispatch_7::accepts_chunk_7(request.operation.as_str()) {
            return self.execute_chunk_7(request);
        }
        if native_boundary_dispatch_8::accepts_chunk_8(request.operation.as_str()) {
            return self.execute_chunk_8(request);
        }
        if native_boundary_dispatch_9::accepts_chunk_9(request.operation.as_str()) {
            return self.execute_chunk_9(request);
        }
        if native_boundary_dispatch_10::accepts_chunk_10(request.operation.as_str()) {
            return self.execute_chunk_10(request);
        }
        if native_boundary_dispatch_11::accepts_chunk_11(request.operation.as_str()) {
            return self.execute_chunk_11(request);
        }
        if native_boundary_dispatch_12::accepts_chunk_12(request.operation.as_str()) {
            return self.execute_chunk_12(request);
        }
        if native_boundary_dispatch_13::accepts_chunk_13(request.operation.as_str()) {
            return self.execute_chunk_13(request);
        }
        if native_boundary_dispatch_14::accepts_chunk_14(request.operation.as_str()) {
            return self.execute_chunk_14(request);
        }
        if native_boundary_dispatch_15::accepts_chunk_15(request.operation.as_str()) {
            return self.execute_chunk_15(request);
        }
        if native_boundary_dispatch_16::accepts_chunk_16(request.operation.as_str()) {
            return self.execute_chunk_16(request);
        }
        if native_boundary_dispatch_17::accepts_chunk_17(request.operation.as_str()) {
            return self.execute_chunk_17(request);
        }
        if native_boundary_dispatch_18::accepts_chunk_18(request.operation.as_str()) {
            return self.execute_chunk_18(request);
        }
        if native_boundary_dispatch_19::accepts_chunk_19(request.operation.as_str()) {
            return self.execute_chunk_19(request);
        }
        if native_boundary_dispatch_20::accepts_chunk_20(request.operation.as_str()) {
            return self.execute_chunk_20(request);
        }
        if native_boundary_dispatch_21::accepts_chunk_21(request.operation.as_str()) {
            return self.execute_chunk_21(request);
        }
        if native_boundary_dispatch_22::accepts_chunk_22(request.operation.as_str()) {
            return self.execute_chunk_22(request);
        }
        if native_boundary_dispatch_23::accepts_chunk_23(request.operation.as_str()) {
            return self.execute_chunk_23(request);
        }
        if native_boundary_dispatch_24::accepts_chunk_24(request.operation.as_str()) {
            return self.execute_chunk_24(request);
        }
        if native_boundary_dispatch_25::accepts_chunk_25(request.operation.as_str()) {
            return self.execute_chunk_25(request);
        }
        if native_boundary_dispatch_26::accepts_chunk_26(request.operation.as_str()) {
            return self.execute_chunk_26(request);
        }
        if native_boundary_dispatch_27::accepts_chunk_27(request.operation.as_str()) {
            return self.execute_chunk_27(request);
        }
        if native_boundary_dispatch_28::accepts_chunk_28(request.operation.as_str()) {
            return self.execute_chunk_28(request);
        }
        if native_boundary_dispatch_29::accepts_chunk_29(request.operation.as_str()) {
            return self.execute_chunk_29(request);
        }
        if native_boundary_dispatch_30::accepts_chunk_30(request.operation.as_str()) {
            return self.execute_chunk_30(request);
        }
        if native_boundary_dispatch_31::accepts_chunk_31(request.operation.as_str()) {
            return self.execute_chunk_31(request);
        }
        if native_boundary_dispatch_32::accepts_chunk_32(request.operation.as_str()) {
            return self.execute_chunk_32(request);
        }
        if native_boundary_dispatch_33::accepts_chunk_33(request.operation.as_str()) {
            return self.execute_chunk_33(request);
        }
        if native_boundary_dispatch_34::accepts_chunk_34(request.operation.as_str()) {
            return self.execute_chunk_34(request);
        }
        if native_boundary_dispatch_35::accepts_chunk_35(request.operation.as_str()) {
            return self.execute_chunk_35(request);
        }
        if native_boundary_dispatch_36::accepts_chunk_36(request.operation.as_str()) {
            return self.execute_chunk_36(request);
        }
        if native_boundary_dispatch_37::accepts_chunk_37(request.operation.as_str()) {
            return self.execute_chunk_37(request);
        }
        if native_boundary_dispatch_38::accepts_chunk_38(request.operation.as_str()) {
            return self.execute_chunk_38(request);
        }
        if native_boundary_dispatch_39::accepts_chunk_39(request.operation.as_str()) {
            return self.execute_chunk_39(request);
        }
        if native_boundary_dispatch_40::accepts_chunk_40(request.operation.as_str()) {
            return self.execute_chunk_40(request);
        }
        if native_boundary_dispatch_41::accepts_chunk_41(request.operation.as_str()) {
            return self.execute_chunk_41(request);
        }
        if native_boundary_dispatch_42::accepts_chunk_42(request.operation.as_str()) {
            return self.execute_chunk_42(request);
        }
        if native_boundary_dispatch_43::accepts_chunk_43(request.operation.as_str()) {
            return self.execute_chunk_43(request);
        }
        if native_boundary_dispatch_44::accepts_chunk_44(request.operation.as_str()) {
            return self.execute_chunk_44(request);
        }
        if native_boundary_dispatch_45::accepts_chunk_45(request.operation.as_str()) {
            return self.execute_chunk_45(request);
        }
        if native_boundary_dispatch_46::accepts_chunk_46(request.operation.as_str()) {
            return self.execute_chunk_46(request);
        }
        if native_boundary_dispatch_47::accepts_chunk_47(request.operation.as_str()) {
            return self.execute_chunk_47(request);
        }
        if native_boundary_dispatch_48::accepts_chunk_48(request.operation.as_str()) {
            return self.execute_chunk_48(request);
        }
        if native_boundary_dispatch_49::accepts_chunk_49(request.operation.as_str()) {
            return self.execute_chunk_49(request);
        }
        if native_boundary_dispatch_50::accepts_chunk_50(request.operation.as_str()) {
            return self.execute_chunk_50(request);
        }
        protocol_error("unknown_operation", &request.operation)
    }

    fn validate(&self, handle: &HandleArg, expected_type: &str) -> Result<(), String> {
        if handle.owner != self.owner {
            return Err(protocol_error(
                "cross_owner_handle",
                "native resource belongs to another worker",
            ));
        }
        if handle.type_name != expected_type {
            return Err(protocol_error("handle_type_mismatch", &handle.type_name));
        }
        match self.handles.get(&handle.id) {
            Some(entry) if entry.generation != handle.generation => {
                Err(protocol_error("stale_handle", "NativeBoundary handle is stale"))
            }
            Some(entry) if entry.value.type_name() != expected_type => Err(protocol_error(
                "handle_storage_mismatch",
                "NativeBoundary handle resource type does not match",
            )),
            Some(_) => Ok(()),
            _ => Err(protocol_error("stale_handle", "NativeBoundary handle is stale")),
        }
    }

    fn live_array(&self, handle: &HandleArg) -> Result<&Array, String> {
        self.validate(handle, "ndarray.Array.Array")?;
        match &self.handles.get(&handle.id).expect("validated handle").value {
            HandleValue::Array(value) => Ok(value),
            _ => Err(protocol_error("handle_storage_mismatch", "ndarray.Array.Array")),
        }
    }

    fn live_array_mut(&mut self, handle: &HandleArg) -> Result<&mut Array, String> {
        self.validate(handle, "ndarray.Array.Array")?;
        match &mut self.handles.get_mut(&handle.id).expect("validated handle").value {
            HandleValue::Array(value) => Ok(value),
            _ => Err(protocol_error("handle_storage_mismatch", "ndarray.Array.Array")),
        }
    }

    fn live_arrowarray(&self, handle: &HandleArg) -> Result<&ArrowArray, String> {
        self.validate(handle, "ndarray.ArrowArray.ArrowArray")?;
        match &self.handles.get(&handle.id).expect("validated handle").value {
            HandleValue::ArrowArray(value) => Ok(value),
            _ => Err(protocol_error("handle_storage_mismatch", "ndarray.ArrowArray.ArrowArray")),
        }
    }

    fn live_randomstream_mut(&mut self, handle: &HandleArg) -> Result<&mut RandomStream, String> {
        self.validate(handle, "ndarray.RandomStream.RandomStream")?;
        match &mut self.handles.get_mut(&handle.id).expect("validated handle").value {
            HandleValue::RandomStream(value) => Ok(value),
            _ => Err(protocol_error("handle_storage_mismatch", "ndarray.RandomStream.RandomStream")),
        }
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
    let _request_id = fields
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
    Ok(Request { operation, args })
}

fn resolve_request_references(
    line: &str,
    request_handles: &HashMap<u64, HandleArg>,
) -> Result<String, String> {
    line.split_whitespace()
        .map(|field| {
            let Some(reference) = field.strip_prefix("r:") else {
                return Ok(field.to_string());
            };
            let parts = reference.split(':').collect::<Vec<_>>();
            let (request_id, overridden_type) = match parts.as_slice() {
                [request_id] => (*request_id, None),
                [request_id, type_name] => (*request_id, Some(*type_name)),
                _ => {
                    return Err(protocol_error(
                        "invalid_handle_reference",
                        "malformed prior-request handle reference",
                    ));
                }
            };
            let request_id = request_id.parse::<u64>().map_err(|error| {
                protocol_error("invalid_handle_reference", &error.to_string())
            })?;
            let handle = request_handles.get(&request_id).ok_or_else(|| {
                protocol_error(
                    "unknown_handle_reference",
                    "prior request did not return a handle",
                )
            })?;
            let type_name = overridden_type
                .map(str::to_string)
                .unwrap_or_else(|| STANDARD.encode(handle.type_name.as_bytes()));
            Ok(format!(
                "h:{}:{}:{}:{type_name}",
                STANDARD.encode(handle.owner.as_bytes()),
                handle.id,
                handle.generation,
            ))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|fields| fields.join(" "))
}

fn response_handle(payload: &str) -> Option<HandleArg> {
    let fields = payload.split_whitespace().collect::<Vec<_>>();
    let ["ok_handle", owner, id, generation, type_name] = fields.as_slice() else {
        return None;
    };
    parse_handle_arg(&format!("{owner}:{id}:{generation}:{type_name}")).ok()
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
    if let Some(value) = value.strip_prefix("x:") {
        let bytes = STANDARD
            .decode(value)
            .map_err(|error| protocol_error("invalid_base64", &error.to_string()))?;
        if bytes.len() > MAX_ADAPTER_TRANSFER_BYTES {
            return Err(protocol_error(
                "transfer_too_large",
                "copied byte argument exceeds the native adapter transfer bound",
            ));
        }
        return Ok(Arg::Bytes(bytes));
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
    if let Some(value) = value.strip_prefix("lh:") {
        if value.is_empty() {
            return Ok(Arg::Handles(Vec::new()));
        }
        return value
            .split(',')
            .map(parse_handle_arg)
            .collect::<Result<Vec<_>, _>>()
            .map(Arg::Handles);
    }
    if let Some(value) = value.strip_prefix("h:") {
        return parse_handle_arg(value).map(Arg::Handle);
    }
    Err(protocol_error("invalid_argument", "unsupported argument encoding"))
}

fn parse_handle_arg(value: &str) -> Result<HandleArg, String> {
    let fields = value.split(':').collect::<Vec<_>>();
    let [owner, id, generation, type_name] = fields.as_slice() else {
        return Err(protocol_error("invalid_argument", "malformed handle"));
    };
    Ok(HandleArg {
        owner: decode_text(owner)?,
        id: id.parse().map_err(|error: std::num::ParseIntError| protocol_error("invalid_argument", &error.to_string()))?,
        generation: generation.parse().map_err(|error: std::num::ParseIntError| protocol_error("invalid_argument", &error.to_string()))?,
        type_name: decode_text(type_name)?,
    })
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

fn arg_handles(value: &Arg) -> &[HandleArg] {
    match value {
        Arg::Handles(values) => values.as_slice(),
        Arg::EmptyList => &[],
        _ => unreachable!("generated argument pattern admits only resource lists"),
    }
}

fn validate_decoded_arg_shape(value: &Arg) {
    match value {
        Arg::Int(value) => { let _ = value; }
        Arg::Float(value) => { let _ = value; }
        Arg::Bool(value) => { let _ = value; }
        Arg::String(value) => { let _ = value; }
        Arg::Bytes(value) => { let _ = value; }
        Arg::Ints(_) => { let _ = arg_ints(value); }
        Arg::Floats(_) => { let _ = arg_floats(value); }
        Arg::Bools(_) => { let _ = arg_bools(value); }
        Arg::Handles(_) => { let _ = arg_handles(value); }
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
