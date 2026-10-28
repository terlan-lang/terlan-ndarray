use super::*;

type Array = *mut crate::ffi::TerlanNdArray;

unsafe fn float_array(values: &[f64], shape: &[i64]) -> Array {
    let mut array = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            crate::ffi::terlan_ndarray_v1_from_floats(
                values.as_ptr(),
                values.len() as i64,
                shape.as_ptr(),
                shape.len() as i64,
                &mut array,
            )
        },
        OK
    );
    array
}

unsafe fn arrays_for_all_dtypes() -> Vec<Array> {
    let shape = [2i64];
    let integers = [-1i64, 1];
    let floats = [-1.5f64, 2.25];
    let bools = [0u8, 1];
    let mut arrays = Vec::new();
    let mut array = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            crate::ffi::terlan_ndarray_v1_from_bools(
                bools.as_ptr(),
                2,
                shape.as_ptr(),
                1,
                &mut array,
            )
        },
        OK
    );
    arrays.push(array);
    for dtype in [2, 4, 5, 6, 7, 8, 9, 10] {
        let values = if matches!(dtype, 7..=10) {
            [0i64, 1]
        } else {
            integers
        };
        array = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                crate::ffi::terlan_ndarray_v1_from_ints_as(
                    values.as_ptr(),
                    2,
                    shape.as_ptr(),
                    1,
                    dtype,
                    &mut array,
                )
            },
            OK
        );
        arrays.push(array);
    }
    for dtype in [3, 11, 14] {
        array = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                crate::ffi::terlan_ndarray_v1_from_floats_as(
                    floats.as_ptr(),
                    2,
                    shape.as_ptr(),
                    1,
                    dtype,
                    &mut array,
                )
            },
            OK
        );
        arrays.push(array);
    }
    for dtype in [12, 13] {
        array = std::ptr::null_mut();
        assert_eq!(
            unsafe {
                crate::ffi::terlan_ndarray_v1_from_complex(
                    floats.as_ptr(),
                    2,
                    floats.as_ptr(),
                    2,
                    shape.as_ptr(),
                    1,
                    dtype,
                    &mut array,
                )
            },
            OK
        );
        arrays.push(array);
    }
    arrays
}

unsafe fn bytes_from(
    operation: unsafe extern "C" fn(Array, *mut *mut u8, *mut usize) -> i32,
    array: Array,
) -> Vec<u8> {
    let mut bytes = std::ptr::null_mut();
    let mut length = 0;
    assert_eq!(unsafe { operation(array, &mut bytes, &mut length) }, OK);
    let output = unsafe { std::slice::from_raw_parts(bytes, length) }.to_vec();
    unsafe { crate::ffi::terlan_ndarray_v1_delete_tensor_packet(bytes) };
    output
}

unsafe fn exact_packet(array: Array) -> Vec<u8> {
    unsafe { export_packet(array).unwrap() }
}

#[test]
fn npy_round_trip_is_deterministic_and_dtype_exact() {
    unsafe {
        let array = float_array(&[1.25, -2.5, 3.75, 4.0], &[2, 2]);
        let first = bytes_from(terlan_ndarray_v1_to_npy, array);
        let second = bytes_from(terlan_ndarray_v1_to_npy, array);
        assert_eq!(first, second);
        assert_eq!(&first[..6], b"\x93NUMPY");
        let mut decoded = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_from_npy(
                first.as_ptr(),
                first.len() as i64,
                first.len() as i64,
                &mut decoded,
            ),
            OK
        );
        assert_eq!(exact_packet(array), exact_packet(decoded));
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(decoded), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(array), OK);
    }
}

#[test]
fn npy_round_trip_preserves_every_admitted_scalar_dtype() {
    unsafe {
        for array in arrays_for_all_dtypes() {
            let bytes = bytes_from(terlan_ndarray_v1_to_npy, array);
            let mut decoded = std::ptr::null_mut();
            assert_eq!(
                terlan_ndarray_v1_from_npy(
                    bytes.as_ptr(),
                    bytes.len() as i64,
                    bytes.len() as i64,
                    &mut decoded,
                ),
                OK
            );
            assert_eq!(exact_packet(array), exact_packet(decoded));
            assert_eq!(crate::ffi::terlan_ndarray_v1_delete(decoded), OK);
            assert_eq!(crate::ffi::terlan_ndarray_v1_delete(array), OK);
        }
    }
}

#[test]
fn raw_round_trip_detects_corruption_and_size_violations() {
    unsafe {
        let array = float_array(&[0.0, 1.0, -1.0], &[3]);
        let bytes = bytes_from(terlan_ndarray_v1_serialize, array);
        let mut decoded = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_deserialize(
                bytes.as_ptr(),
                bytes.len() as i64,
                bytes.len() as i64,
                &mut decoded,
            ),
            OK
        );
        assert_eq!(exact_packet(array), exact_packet(decoded));
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(decoded), OK);
        let mut corrupt = bytes.clone();
        *corrupt.last_mut().unwrap() ^= 0xff;
        assert_eq!(
            terlan_ndarray_v1_deserialize(
                corrupt.as_ptr(),
                corrupt.len() as i64,
                corrupt.len() as i64,
                &mut decoded,
            ),
            INTEGRITY_ERROR
        );
        assert!(decoded.is_null());
        assert_eq!(
            terlan_ndarray_v1_deserialize(
                bytes.as_ptr(),
                bytes.len() as i64,
                (bytes.len() - 1) as i64,
                &mut decoded,
            ),
            INVALID_ARGUMENT
        );
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(array), OK);
    }
}

#[test]
fn npy_decode_rejects_malformed_truncated_and_endian_inputs() {
    unsafe {
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_from_npy(b"bad".as_ptr(), 3, 3, &mut output),
            MALFORMED_DATA
        );
        let array = float_array(&[1.0, 2.0], &[2]);
        let mut bytes = bytes_from(terlan_ndarray_v1_to_npy, array);
        bytes.pop();
        assert_eq!(
            terlan_ndarray_v1_from_npy(
                bytes.as_ptr(),
                bytes.len() as i64,
                bytes.len() as i64,
                &mut output,
            ),
            MALFORMED_DATA
        );
        let mut endian = bytes_from(terlan_ndarray_v1_to_npy, array);
        let location = endian.windows(3).position(|value| value == b"<f8").unwrap();
        endian[location] = b'>';
        assert_eq!(
            terlan_ndarray_v1_from_npy(
                endian.as_ptr(),
                endian.len() as i64,
                endian.len() as i64,
                &mut output,
            ),
            UNSUPPORTED_LAYOUT
        );
        assert!(output.is_null());
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(array), OK);
    }
}

#[test]
fn serialization_rejects_null_outputs_and_invalid_bounds() {
    unsafe {
        let array = float_array(&[1.0], &[1]);
        let mut length = 0;
        assert_eq!(
            terlan_ndarray_v1_to_npy(array, std::ptr::null_mut(), &mut length),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_serialize(array, std::ptr::null_mut(), &mut length),
            INVALID_ARGUMENT
        );
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_from_npy(std::ptr::null(), -1, 1, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_deserialize(std::ptr::null(), 1, 1, &mut output),
            INVALID_ARGUMENT
        );
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(array), OK);
    }
}

#[test]
fn mmap_preserves_read_only_and_flushes_writable_storage() {
    unsafe {
        let source = float_array(&[1.0, 2.0, 3.0], &[3]);
        let bytes = bytes_from(terlan_ndarray_v1_to_npy, source);
        let path = std::env::temp_dir().join(format!(
            "terlan-ndarray-mmap-{}-{}.npy",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let path_text = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let mut read_only = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_mmap_npy(path_text.as_ptr(), 0, bytes.len() as i64, &mut read_only,),
            OK
        );
        let index = [1i64];
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_float(read_only, index.as_ptr(), 1, 7.0,),
            121
        );
        assert_eq!(crate::ffi::terlan_ndarray_v1_flush(read_only), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(read_only), OK);

        let mut writable = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_mmap_npy(path_text.as_ptr(), 1, bytes.len() as i64, &mut writable,),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_float(writable, index.as_ptr(), 1, 7.0,),
            OK
        );
        assert_eq!(crate::ffi::terlan_ndarray_v1_flush(writable), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(writable), OK);

        assert_eq!(
            terlan_ndarray_v1_mmap_npy(path_text.as_ptr(), 0, bytes.len() as i64, &mut read_only,),
            OK
        );
        let mut value = 0.0;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_get_float(read_only, index.as_ptr(), 1, &mut value,),
            OK
        );
        assert_eq!(value, 7.0);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(read_only), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(source), OK);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn npy_round_trip_preserves_fortran_order_and_logical_values() {
    unsafe {
        let source = float_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mut fortran = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(source, 1, &mut fortran),
            OK
        );
        let bytes = bytes_from(terlan_ndarray_v1_to_npy, fortran);
        assert!(bytes.windows(4).any(|window| window == b"True"));
        let mut decoded = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_from_npy(
                bytes.as_ptr(),
                bytes.len() as i64,
                bytes.len() as i64,
                &mut decoded,
            ),
            OK
        );
        let mut c_contiguous = -1;
        let mut f_contiguous = -1;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_contiguous(decoded, &mut c_contiguous),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(decoded, &mut f_contiguous,),
            OK
        );
        assert_eq!((c_contiguous, f_contiguous), (0, 1));
        assert_eq!(exact_packet(source), exact_packet(decoded));
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(decoded), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(fortran), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(source), OK);
    }
}

#[test]
fn path_save_and_bounded_load_preserve_numpy_files() {
    unsafe {
        let source = float_array(&[1.0, -2.0, 3.5, 4.25], &[2, 2]);
        let path = std::env::temp_dir().join(format!(
            "terlan-ndarray-save-load-{}-{}.npy",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path_text = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();
        assert_eq!(terlan_ndarray_v1_save_npy(source, path_text.as_ptr()), OK);
        let file_length = std::fs::metadata(&path).unwrap().len() as i64;
        let mut decoded = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_load_npy(path_text.as_ptr(), file_length, &mut decoded),
            OK
        );
        assert_eq!(exact_packet(source), exact_packet(decoded));
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(decoded), OK);
        assert_eq!(
            terlan_ndarray_v1_load_npy(path_text.as_ptr(), file_length - 1, &mut decoded),
            INVALID_ARGUMENT
        );
        assert!(decoded.is_null());
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(source), OK);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn fortran_mmap_uses_file_order_for_reads_and_writes() {
    unsafe {
        let source = float_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mut fortran = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_copy_ordered(source, 1, &mut fortran),
            OK
        );
        let bytes = bytes_from(terlan_ndarray_v1_to_npy, fortran);
        let path = std::env::temp_dir().join(format!(
            "terlan-ndarray-fortran-mmap-{}-{}.npy",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let path_text = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let mut mapped = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_mmap_npy(path_text.as_ptr(), 1, bytes.len() as i64, &mut mapped),
            OK
        );
        let index = [1i64, 0i64];
        let mut value = 0.0;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_get_float(mapped, index.as_ptr(), 2, &mut value),
            OK
        );
        assert_eq!(value, 4.0);
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_float(mapped, index.as_ptr(), 2, 40.0),
            OK
        );
        assert_eq!(crate::ffi::terlan_ndarray_v1_flush(mapped), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(mapped), OK);
        let mut decoded = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_load_npy(path_text.as_ptr(), bytes.len() as i64, &mut decoded),
            OK
        );
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_get_float(decoded, index.as_ptr(), 2, &mut value),
            OK
        );
        assert_eq!(value, 40.0);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(decoded), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(fortran), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(source), OK);
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn path_serialization_rejects_invalid_paths_and_malformed_files() {
    unsafe {
        let source = float_array(&[1.0], &[1]);
        let mut output = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_save_npy(source, std::ptr::null()),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_load_npy(std::ptr::null(), 1, &mut output),
            INVALID_ARGUMENT
        );
        let missing = std::ffi::CString::new("/definitely/missing/terlan-array.npy").unwrap();
        assert_eq!(
            terlan_ndarray_v1_load_npy(missing.as_ptr(), 1024, &mut output),
            IO_ERROR
        );
        assert!(output.is_null());
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(source), OK);
    }
}

#[test]
fn mapped_views_keep_storage_alive_and_flush_after_base_disposal() {
    unsafe {
        let source = float_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let bytes = bytes_from(terlan_ndarray_v1_to_npy, source);
        let path = std::env::temp_dir().join(format!(
            "terlan-ndarray-mapped-view-{}-{}.npy",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let path_text = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let mut mapped = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_mmap_npy(path_text.as_ptr(), 1, bytes.len() as i64, &mut mapped),
            OK
        );
        let mut view = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_slice_axis(mapped, 0, 0, 1, 1, &mut view),
            OK
        );
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(mapped), OK);
        let index = [0i64, 2i64];
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_set_float(view, index.as_ptr(), 2, 30.0),
            OK
        );
        assert_eq!(crate::ffi::terlan_ndarray_v1_flush(view), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(view), OK);
        let mut loaded = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_load_npy(path_text.as_ptr(), bytes.len() as i64, &mut loaded),
            OK
        );
        let mut value = 0.0;
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_get_float(loaded, index.as_ptr(), 2, &mut value),
            OK
        );
        assert_eq!(value, 30.0);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(loaded), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(source), OK);
        std::fs::remove_file(path).unwrap();
    }
}

fn version_two_npy(version_one: &[u8]) -> Vec<u8> {
    let header_length = u16::from_le_bytes([version_one[8], version_one[9]]) as usize;
    let header = &version_one[10..10 + header_length];
    let dictionary_end = header
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .unwrap()
        + 1;
    let dictionary = &header[..dictionary_end];
    let padding = (16 - ((12 + dictionary.len() + 1) % 16)) % 16;
    let version_two_header = dictionary.len() + padding + 1;
    let mut output = Vec::new();
    output.extend_from_slice(b"\x93NUMPY\x02\x00");
    output.extend_from_slice(&(version_two_header as u32).to_le_bytes());
    output.extend_from_slice(dictionary);
    output.resize(12 + version_two_header - 1, b' ');
    output.push(b'\n');
    output.extend_from_slice(&version_one[10 + header_length..]);
    output
}

#[test]
fn numerical_version_two_headers_decode_with_aligned_payloads() {
    unsafe {
        let source = float_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let version_one = bytes_from(terlan_ndarray_v1_to_npy, source);
        let version_two = version_two_npy(&version_one);
        assert_eq!(&version_two[..8], b"\x93NUMPY\x02\x00");
        assert_eq!(npy_payload_offset(&version_two).unwrap() % 16, 0);
        let mut decoded = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_from_npy(
                version_two.as_ptr(),
                version_two.len() as i64,
                version_two.len() as i64,
                &mut decoded,
            ),
            OK
        );
        assert_eq!(exact_packet(source), exact_packet(decoded));
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(decoded), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(source), OK);
    }
}
