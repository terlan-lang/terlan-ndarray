#include "../bindings/terlan_ndarray_internal.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void test_round_trip_is_owned_and_preserves_source(void) {
  const double values[] = {1.5, 2.5, 3.5, 4.5};
  const int64_t shape[] = {2, 2};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle imported = NULL;
  DLManagedTensorVersioned *exchange = NULL;
  double *copy = NULL;
  size_t length = 0;
  const int64_t cleanup_before = terlan_ndarray_v1_dlpack_deleter_count();

  assert(terlan_ndarray_v1_from_floats(values, 4, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_to_dlpack(source, &exchange) ==
         TERLAN_NDARRAY_OK);
  assert(exchange != NULL);
  assert(exchange->version.major == 1);
  assert(exchange->version.minor == 3);
  assert(exchange->dl_tensor.device.device_type == kDLCPU);
  assert(exchange->dl_tensor.dtype.code == kDLFloat);
  assert(exchange->dl_tensor.dtype.bits == 64);
  assert(terlan_ndarray_v1_from_dlpack(exchange, &imported) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dlpack_deleter_count() == cleanup_before + 1);

  ((double *)source->data)[0] = 99.0;
  assert(terlan_ndarray_v1_to_floats(imported, &copy, &length) ==
         TERLAN_NDARRAY_OK);
  assert(length == 4);
  assert(copy[0] == 1.5);
  terlan_ndarray_v1_delete_float_values(copy);
  assert(terlan_ndarray_v1_delete(imported) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

typedef struct AdversarialTensor {
  DLManagedTensorVersioned managed;
  int64_t shape[2];
  int64_t strides[2];
  double values[4];
  int *cleanup_count;
} AdversarialTensor;

static void adversarial_deleter(DLManagedTensorVersioned *managed) {
  AdversarialTensor *fixture = managed->manager_ctx;
  *fixture->cleanup_count += 1;
}

static AdversarialTensor fixture(int *cleanup_count) {
  AdversarialTensor result = {0};
  result.shape[0] = 2;
  result.shape[1] = 2;
  result.strides[0] = 2;
  result.strides[1] = 1;
  result.values[0] = 1.0;
  result.values[1] = 2.0;
  result.values[2] = 3.0;
  result.values[3] = 4.0;
  result.cleanup_count = cleanup_count;
  result.managed.version.major = DLPACK_MAJOR_VERSION;
  result.managed.version.minor = DLPACK_MINOR_VERSION;
  result.managed.manager_ctx = &result;
  result.managed.deleter = adversarial_deleter;
  result.managed.dl_tensor.data = result.values;
  result.managed.dl_tensor.device.device_type = kDLCPU;
  result.managed.dl_tensor.device.device_id = 0;
  result.managed.dl_tensor.ndim = 2;
  result.managed.dl_tensor.dtype.code = kDLFloat;
  result.managed.dl_tensor.dtype.bits = 64;
  result.managed.dl_tensor.dtype.lanes = 1;
  result.managed.dl_tensor.shape = result.shape;
  result.managed.dl_tensor.strides = result.strides;
  return result;
}

static void repair_self_pointers(AdversarialTensor *value) {
  value->managed.manager_ctx = value;
  value->managed.dl_tensor.data = value->values;
  value->managed.dl_tensor.shape = value->shape;
  value->managed.dl_tensor.strides = value->strides;
}

static void test_rejection_paths_invoke_deleter_once(void) {
  int cleanup_count = 0;
  AdversarialTensor value = fixture(&cleanup_count);
  TerlanNdArrayHandle imported = NULL;

  repair_self_pointers(&value);
  value.managed.version.minor = DLPACK_MINOR_VERSION + 1;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_OK);
  assert(cleanup_count == 1);
  assert(terlan_ndarray_v1_delete(imported) == TERLAN_NDARRAY_OK);
  imported = NULL;

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.flags =
      DLPACK_FLAG_BITMASK_READ_ONLY | DLPACK_FLAG_BITMASK_IS_COPIED;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_OK);
  assert(cleanup_count == 2);
  assert(terlan_ndarray_v1_delete(imported) == TERLAN_NDARRAY_OK);
  imported = NULL;

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.dl_tensor.strides = NULL;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_OK);
  assert(cleanup_count == 3);
  assert(terlan_ndarray_v1_delete(imported) == TERLAN_NDARRAY_OK);
  imported = NULL;

  repair_self_pointers(&value);
  value.managed.version.major = 2;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_VERSION);
  assert(cleanup_count == 4);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.flags = DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_LAYOUT);
  assert(cleanup_count == 5);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.dl_tensor.device.device_type = kDLCUDA;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_DEVICE);
  assert(cleanup_count == 6);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.dl_tensor.device.device_id = 1;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_DEVICE);
  assert(cleanup_count == 7);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.dl_tensor.dtype.bits = 24;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_DTYPE);
  assert(cleanup_count == 8);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.dl_tensor.dtype.lanes = 2;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_DTYPE);
  assert(cleanup_count == 9);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.dl_tensor.ndim = -1;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_LAYOUT);
  assert(cleanup_count == 10);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.dl_tensor.ndim = 33;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_LAYOUT);
  assert(cleanup_count == 11);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.dl_tensor.shape = NULL;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_LAYOUT);
  assert(cleanup_count == 12);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.dl_tensor.byte_offset = 1;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_LAYOUT);
  assert(cleanup_count == 13);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.shape[0] = -1;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_NEGATIVE_DIMENSION);
  assert(cleanup_count == 14);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.shape[0] = INT64_MAX;
  value.shape[1] = 2;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_SIZE_OVERFLOW);
  assert(cleanup_count == 15);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.strides[0] = 1;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_LAYOUT);
  assert(cleanup_count == 16);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.dl_tensor.data = NULL;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_LAYOUT);
  assert(cleanup_count == 17);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.dl_tensor.data = (uint8_t *)value.values + 1;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_DLPACK_LAYOUT);
  assert(cleanup_count == 18);

  value = fixture(&cleanup_count);
  repair_self_pointers(&value);
  value.managed.dl_tensor.dtype.code = kDLBool;
  value.managed.dl_tensor.dtype.bits = 8;
  ((uint8_t *)value.values)[0] = 2;
  assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
         TERLAN_NDARRAY_INVALID_BOOL);
  assert(cleanup_count == 19);
}

static void test_zero_dimension_round_trip_uses_canonical_strides(void) {
  const int64_t shape[] = {2, 0, 3};
  const int64_t expected_strides[] = {0, 3, 1};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle imported = NULL;
  DLManagedTensorVersioned *exchange = NULL;
  int64_t *strides = NULL;

  assert(terlan_ndarray_v1_from_ints(NULL, 0, shape, 3, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_to_dlpack(source, &exchange) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_from_dlpack(exchange, &imported) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_strides(imported, &strides) == TERLAN_NDARRAY_OK);
  assert(strides[0] == expected_strides[0]);
  assert(strides[1] == expected_strides[1]);
  assert(strides[2] == expected_strides[2]);
  assert(terlan_ndarray_v1_delete(imported) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_pointer_free_tensor_packet_uses_dlpack_lifecycle(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle imported = NULL;
  uint8_t *packet = NULL;
  size_t packet_length = 0;
  int64_t *copy = NULL;
  size_t copy_length = 0;
  const int64_t cleanup_before = terlan_ndarray_v1_dlpack_deleter_count();

  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_export_tensor_packet(source, "pytorch", &packet,
                                                &packet_length) ==
         TERLAN_NDARRAY_OK);
  assert(packet_length > 16);
  assert(terlan_ndarray_v1_dlpack_deleter_count() == cleanup_before + 1);
  assert(terlan_ndarray_v1_import_tensor_packet(packet, (int64_t)packet_length,
                                                &imported) ==
         TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_dlpack_deleter_count() == cleanup_before + 2);
  assert(terlan_ndarray_v1_to_ints(imported, &copy, &copy_length) ==
         TERLAN_NDARRAY_OK);
  assert(copy_length == 6);
  assert(copy[5] == 6);
  terlan_ndarray_v1_delete_int_values(copy);
  terlan_ndarray_v1_delete_tensor_packet(packet);
  assert(terlan_ndarray_v1_delete(imported) == TERLAN_NDARRAY_OK);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

static void test_allocation_and_packet_failures_release_every_phase(void) {
  const int64_t values[] = {1, 2, 3, 4, 5, 6};
  const int64_t shape[] = {2, 3};
  TerlanNdArrayHandle source = NULL;
  TerlanNdArrayHandle imported = NULL;
  DLManagedTensorVersioned *exchange = NULL;
  uint8_t *packet = NULL;
  uint8_t *trailing = NULL;
  size_t packet_length = 0;
  int64_t fail_after;
  int64_t cleanup_before;
  int cleanup_count = 0;

  assert(terlan_ndarray_v1_from_ints(values, 6, shape, 2, &source) ==
         TERLAN_NDARRAY_OK);
  cleanup_before = terlan_ndarray_v1_dlpack_deleter_count();
  for (fail_after = 0; fail_after < 4; ++fail_after) {
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_to_dlpack(source, &exchange) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(exchange == NULL);
    terlan_ndarray_test_reset_allocation_failures();
  }
  assert(terlan_ndarray_v1_dlpack_deleter_count() == cleanup_before);

  for (fail_after = 0; fail_after < 4; ++fail_after) {
    AdversarialTensor value = fixture(&cleanup_count);
    repair_self_pointers(&value);
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_from_dlpack(&value.managed, &imported) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(imported == NULL);
    terlan_ndarray_test_reset_allocation_failures();
    assert(cleanup_count == fail_after + 1);
  }

  assert(terlan_ndarray_v1_export_tensor_packet(source, "pytorch", &packet,
                                                &packet_length) ==
         TERLAN_NDARRAY_OK);
  cleanup_before = terlan_ndarray_v1_dlpack_deleter_count();
  for (fail_after = 0; fail_after < 8; ++fail_after) {
    terlan_ndarray_test_fail_allocation_after(fail_after);
    assert(terlan_ndarray_v1_import_tensor_packet(packet,
                                                  (int64_t)packet_length,
                                                  &imported) ==
           TERLAN_NDARRAY_ALLOCATION_FAILED);
    assert(imported == NULL);
    terlan_ndarray_test_reset_allocation_failures();
  }
  assert(terlan_ndarray_v1_dlpack_deleter_count() == cleanup_before + 4);

  assert(terlan_ndarray_v1_import_tensor_packet(packet,
                                                (int64_t)packet_length - 1,
                                                &imported) ==
         TERLAN_NDARRAY_DLPACK_DTYPE);
  assert(imported == NULL);
  trailing = malloc(packet_length + 1);
  assert(trailing != NULL);
  memcpy(trailing, packet, packet_length);
  trailing[packet_length] = 0;
  assert(terlan_ndarray_v1_import_tensor_packet(
             trailing, (int64_t)packet_length + 1, &imported) ==
         TERLAN_NDARRAY_DLPACK_DTYPE);
  assert(imported == NULL);
  memcpy(trailing, "BAD!", 4);
  assert(terlan_ndarray_v1_import_tensor_packet(
             trailing, (int64_t)packet_length, &imported) ==
         TERLAN_NDARRAY_DLPACK_VERSION);
  assert(imported == NULL);

  free(trailing);
  terlan_ndarray_v1_delete_tensor_packet(packet);
  assert(terlan_ndarray_v1_delete(source) == TERLAN_NDARRAY_OK);
}

int main(void) {
  test_round_trip_is_owned_and_preserves_source();
  test_rejection_paths_invoke_deleter_once();
  test_zero_dimension_round_trip_uses_canonical_strides();
  test_pointer_free_tensor_packet_uses_dlpack_lifecycle();
  test_allocation_and_packet_failures_release_every_phase();
  assert(terlan_ndarray_v1_live_count() == 0);
  return 0;
}
