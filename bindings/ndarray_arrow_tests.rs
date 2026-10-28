use super::*;
use arrow::array::{Array, Float32Array, Int16Array};

unsafe fn floats(values: &[f64], shape: &[i64]) -> *mut crate::ffi::TerlanNdArray {
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

#[test]
fn arrow_export_owns_callbacks_and_round_trips_shape_and_values() {
    unsafe {
        let source = floats(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let mut arrow = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_to_arrow(source, &mut arrow), OK);
        assert!(!arrow.is_null());
        assert!(!terlan_ndarray_v1_arrow_array_pointer(arrow).is_null());
        assert!(!terlan_ndarray_v1_arrow_schema_pointer(arrow).is_null());
        let mut length = -1;
        let mut null_count = -1;
        assert_eq!(terlan_ndarray_v1_arrow_length(arrow, &mut length), OK);
        assert_eq!(
            terlan_ndarray_v1_arrow_null_count(arrow, &mut null_count),
            OK
        );
        assert_eq!((length, null_count), (4, 0));
        let mut decoded = std::ptr::null_mut();
        assert_eq!(terlan_ndarray_v1_from_arrow(arrow, &mut decoded), OK);
        assert_eq!(
            export_packet(source).unwrap(),
            export_packet(decoded).unwrap()
        );
        assert_eq!(terlan_ndarray_v1_arrow_delete(arrow), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(decoded), OK);
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(source), OK);
    }
}

#[test]
fn arrow_export_rejects_complex_and_malformed_outputs() {
    unsafe {
        let real = [1.0];
        let imaginary = [2.0];
        let shape = [1i64];
        let mut complex = std::ptr::null_mut();
        assert_eq!(
            crate::ffi::terlan_ndarray_v1_from_complex(
                real.as_ptr(),
                1,
                imaginary.as_ptr(),
                1,
                shape.as_ptr(),
                1,
                13,
                &mut complex,
            ),
            OK
        );
        let mut arrow = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_to_arrow(complex, &mut arrow),
            UNSUPPORTED_DTYPE
        );
        assert!(arrow.is_null());
        assert_eq!(
            terlan_ndarray_v1_to_arrow(complex, std::ptr::null_mut()),
            INVALID_ARGUMENT
        );
        assert_eq!(
            terlan_ndarray_v1_from_arrow(std::ptr::null_mut(), &mut complex),
            INVALID_ARGUMENT
        );
        assert!(terlan_ndarray_v1_arrow_array_pointer(std::ptr::null_mut()).is_null());
        assert!(terlan_ndarray_v1_arrow_schema_pointer(std::ptr::null_mut()).is_null());
        assert_eq!(
            terlan_ndarray_v1_arrow_delete(std::ptr::null_mut()),
            INVALID_ARGUMENT
        );
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(complex), OK);
    }
}

#[test]
fn external_arrow_c_data_moves_ownership_and_preserves_exact_values() {
    unsafe {
        let data = Int16Array::from(vec![-32768, -1, 0, 32767]).to_data();
        let (mut array, mut schema) = to_ffi(&data).unwrap();
        let shape = [2i64, 2];
        let mut imported = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_import_arrow_c(
                (&mut array as *mut FFI_ArrowArray).cast(),
                (&mut schema as *mut FFI_ArrowSchema).cast(),
                shape.as_ptr(),
                shape.len() as i64,
                &mut imported,
            ),
            OK
        );
        assert!(array.is_released());
        let packet = parse_tensor_packet(&export_packet(imported).unwrap()).unwrap();
        assert_eq!((packet.dtype_code, packet.bits), (0, 16));
        assert_eq!(packet.shape, vec![2, 2]);
        assert_eq!(
            packet.data,
            [-32768i16, -1, 0, 32767].map(i16::to_le_bytes).concat()
        );
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(imported), OK);
    }
}

#[test]
fn external_arrow_import_respects_nonzero_primitive_offsets() {
    unsafe {
        let values = Int16Array::from(vec![100, 200, 300, 400]);
        let data = values.slice(1, 2).to_data();
        let (mut array, mut schema) = to_ffi(&data).unwrap();
        let shape = [2i64];
        let mut imported = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_import_arrow_c(
                (&mut array as *mut FFI_ArrowArray).cast(),
                (&mut schema as *mut FFI_ArrowSchema).cast(),
                shape.as_ptr(),
                1,
                &mut imported,
            ),
            OK
        );
        let packet = parse_tensor_packet(&export_packet(imported).unwrap()).unwrap();
        assert_eq!(packet.data, [200i16, 300].map(i16::to_le_bytes).concat());
        assert_eq!(crate::ffi::terlan_ndarray_v1_delete(imported), OK);
    }
}

#[test]
fn external_arrow_import_validates_shape_nulls_and_argument_ownership() {
    unsafe {
        let data = Float32Array::from(vec![Some(1.0), None]).to_data();
        let (mut array, mut schema) = to_ffi(&data).unwrap();
        let shape = [2i64];
        let mut imported = std::ptr::null_mut();
        assert_eq!(
            terlan_ndarray_v1_import_arrow_c(
                (&mut array as *mut FFI_ArrowArray).cast(),
                (&mut schema as *mut FFI_ArrowSchema).cast(),
                shape.as_ptr(),
                1,
                &mut imported,
            ),
            UNSUPPORTED_DTYPE
        );
        assert!(array.is_released());
        assert!(imported.is_null());

        let data = Float32Array::from(vec![1.0, 2.0]).to_data();
        let (mut array, mut schema) = to_ffi(&data).unwrap();
        let bad_shape = [3i64];
        assert_eq!(
            terlan_ndarray_v1_import_arrow_c(
                (&mut array as *mut FFI_ArrowArray).cast(),
                (&mut schema as *mut FFI_ArrowSchema).cast(),
                bad_shape.as_ptr(),
                1,
                &mut imported,
            ),
            MALFORMED_DATA
        );
        assert!(array.is_released());

        assert_eq!(
            terlan_ndarray_v1_import_arrow_c(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                &mut imported,
            ),
            INVALID_ARGUMENT
        );
    }
}
