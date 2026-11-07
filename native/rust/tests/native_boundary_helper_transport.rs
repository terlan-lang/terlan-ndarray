use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

const FRAME_BYTES: usize = 1_048_576;

#[test]
fn helper_accepts_a_chunked_request_larger_than_one_mebibyte() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_native-boundary-helper"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start ndarray native helper");
    let mut input = child.stdin.take().expect("helper stdin");
    let output = child.stdout.take().expect("helper stdout");
    let message = format!(
        "call 1 {} x:{}",
        STANDARD.encode("ndarray.array.import_tensor_packet"),
        STANDARD.encode(vec![0_u8; 900_000]),
    );
    assert!(message.len() > FRAME_BYTES);

    let chunk_bytes = (FRAME_BYTES - 128) * 3 / 4;
    let chunks = message.as_bytes().chunks(chunk_bytes);
    let count = chunks.len();
    for (index, chunk) in chunks.enumerate() {
        writeln!(
            input,
            "chunk 1 {index} {} {}",
            usize::from(index + 1 == count),
            STANDARD.encode(chunk),
        )
        .expect("write request chunk");
    }
    input.flush().expect("flush request chunks");

    let mut reply = String::new();
    BufReader::new(output)
        .read_line(&mut reply)
        .expect("read correlated helper reply");
    assert!(reply.starts_with("reply 1 1 err "), "{reply}");

    drop(input);
    assert!(child.wait().expect("wait for helper").success());
}
