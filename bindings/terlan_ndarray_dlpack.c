#include "terlan_ndarray_internal.h"

#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TERLAN_NDARRAY_DLPACK_MAX_RANK 32
#define TERLAN_TENSOR_PACKET_HEADER_SIZE 16

typedef struct TerlanNdArrayDLPackManager {
  DLManagedTensorVersioned tensor;
} TerlanNdArrayDLPackManager;

static _Atomic int64_t dlpack_deleter_count = 0;

static void delete_managed_tensor(DLManagedTensorVersioned *tensor) {
  TerlanNdArrayDLPackManager *manager;
  if (tensor == NULL) {
    return;
  }
  manager = tensor->manager_ctx;
  if (manager == NULL) {
    return;
  }
  free(manager->tensor.dl_tensor.data);
  free(manager->tensor.dl_tensor.strides);
  free(manager->tensor.dl_tensor.shape);
  atomic_fetch_add_explicit(&dlpack_deleter_count, 1, memory_order_relaxed);
  free(manager);
}

static DLDataType dlpack_dtype(int32_t dtype) {
  DLDataType result = {0, 0, 0};
  const size_t element_size = dtype_size(dtype);
  result.lanes = 1;
  switch (dtype) {
  case TERLAN_NDARRAY_DTYPE_BOOL:
    result.code = (uint8_t)kDLBool;
    break;
  case TERLAN_NDARRAY_DTYPE_INT8:
  case TERLAN_NDARRAY_DTYPE_INT16:
  case TERLAN_NDARRAY_DTYPE_INT32:
  case TERLAN_NDARRAY_DTYPE_INT64:
    result.code = (uint8_t)kDLInt;
    break;
  case TERLAN_NDARRAY_DTYPE_UINT8:
  case TERLAN_NDARRAY_DTYPE_UINT16:
  case TERLAN_NDARRAY_DTYPE_UINT32:
  case TERLAN_NDARRAY_DTYPE_UINT64:
    result.code = (uint8_t)kDLUInt;
    break;
  case TERLAN_NDARRAY_DTYPE_FLOAT16:
  case TERLAN_NDARRAY_DTYPE_FLOAT32:
  case TERLAN_NDARRAY_DTYPE_FLOAT64:
    result.code = (uint8_t)kDLFloat;
    break;
  case TERLAN_NDARRAY_DTYPE_COMPLEX64:
  case TERLAN_NDARRAY_DTYPE_COMPLEX128:
    result.code = (uint8_t)kDLComplex;
    break;
  default:
    result.lanes = 0;
    break;
  }
  if (result.lanes != 0) {
    result.bits = (uint8_t)(element_size * 8);
  }
  return result;
}

static int32_t ndarray_dtype(DLDataType dtype) {
  if (dtype.lanes != 1) {
    return 0;
  }
  if (dtype.code == (uint8_t)kDLBool) {
    return dtype.bits == 8 ? TERLAN_NDARRAY_DTYPE_BOOL : 0;
  }
  if (dtype.code == (uint8_t)kDLInt) {
    switch (dtype.bits) {
    case 8:
      return TERLAN_NDARRAY_DTYPE_INT8;
    case 16:
      return TERLAN_NDARRAY_DTYPE_INT16;
    case 32:
      return TERLAN_NDARRAY_DTYPE_INT32;
    case 64:
      return TERLAN_NDARRAY_DTYPE_INT64;
    default:
      return 0;
    }
  }
  if (dtype.code == (uint8_t)kDLUInt) {
    switch (dtype.bits) {
    case 8:
      return TERLAN_NDARRAY_DTYPE_UINT8;
    case 16:
      return TERLAN_NDARRAY_DTYPE_UINT16;
    case 32:
      return TERLAN_NDARRAY_DTYPE_UINT32;
    case 64:
      return TERLAN_NDARRAY_DTYPE_UINT64;
    default:
      return 0;
    }
  }
  if (dtype.code == (uint8_t)kDLFloat) {
    return dtype.bits == 16 ? TERLAN_NDARRAY_DTYPE_FLOAT16
                            : dtype.bits == 32 ? TERLAN_NDARRAY_DTYPE_FLOAT32
                            : (dtype.bits == 64
                                   ? TERLAN_NDARRAY_DTYPE_FLOAT64
                                   : 0);
  }
  if (dtype.code == (uint8_t)kDLComplex) {
    return dtype.bits == 64 ? TERLAN_NDARRAY_DTYPE_COMPLEX64
                            : (dtype.bits == 128
                                   ? TERLAN_NDARRAY_DTYPE_COMPLEX128
                                   : 0);
  }
  return 0;
}

static void destroy_export(TerlanNdArrayDLPackManager *manager) {
  if (manager == NULL) {
    return;
  }
  free(manager->tensor.dl_tensor.data);
  free(manager->tensor.dl_tensor.strides);
  free(manager->tensor.dl_tensor.shape);
  free(manager);
}

TerlanNdArrayStatus terlan_ndarray_v1_to_dlpack(
    TerlanNdArrayHandle array, DLManagedTensorVersioned **out_tensor) {
  TerlanNdArrayDLPackManager *manager;
  DLTensor *tensor;
  size_t dimensions_size;
  TerlanNdArrayStatus status = require_live(array);

  if (status != TERLAN_NDARRAY_OK || out_tensor == NULL) {
    return status != TERLAN_NDARRAY_OK ? status
                                       : TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_tensor = NULL;
  if (array->rank > INT32_MAX ||
      (uint64_t)array->rank > SIZE_MAX / sizeof(int64_t)) {
    return TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  manager = checked_calloc(1, sizeof(*manager));
  if (manager == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  tensor = &manager->tensor.dl_tensor;
  dimensions_size = (size_t)array->rank * sizeof(int64_t);
  if (array->rank > 0) {
    int64_t stride = 1;
    int64_t dimension;
    tensor->shape = checked_malloc(dimensions_size);
    tensor->strides = checked_malloc(dimensions_size);
    if (tensor->shape == NULL || tensor->strides == NULL) {
      destroy_export(manager);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    memcpy(tensor->shape, array->shape, dimensions_size);
    for (dimension = array->rank; dimension > 0; --dimension) {
      const int64_t index = dimension - 1;
      tensor->strides[index] = stride;
      stride *= array->shape[index];
    }
  }
  if (array->byte_count > 0) {
    int64_t index;
    const size_t element_size = dtype_size(array->dtype);
    tensor->data = checked_malloc((size_t)array->byte_count);
    if (tensor->data == NULL) {
      destroy_export(manager);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    for (index = 0; index < array->numel; ++index) {
      memcpy((uint8_t *)tensor->data + (size_t)index * element_size,
             array_const_element_pointer(array, index), element_size);
    }
  }
  tensor->device.device_type = kDLCPU;
  tensor->device.device_id = 0;
  tensor->ndim = (int32_t)array->rank;
  tensor->dtype = dlpack_dtype(array->dtype);
  tensor->byte_offset = 0;
  manager->tensor.version.major = DLPACK_MAJOR_VERSION;
  manager->tensor.version.minor = DLPACK_MINOR_VERSION;
  manager->tensor.manager_ctx = manager;
  manager->tensor.deleter = delete_managed_tensor;
  manager->tensor.flags =
      DLPACK_FLAG_BITMASK_READ_ONLY | DLPACK_FLAG_BITMASK_IS_COPIED;
  *out_tensor = &manager->tensor;
  return TERLAN_NDARRAY_OK;
}

static TerlanNdArrayStatus validate_import(const DLTensor *tensor,
                                            int32_t *out_dtype,
                                            int64_t *out_numel) {
  int64_t index;
  int64_t numel = 1;
  int64_t expected_stride = 1;
  size_t element_size;

  if (tensor->device.device_type != kDLCPU || tensor->device.device_id != 0) {
    return TERLAN_NDARRAY_DLPACK_DEVICE;
  }
  *out_dtype = ndarray_dtype(tensor->dtype);
  if (*out_dtype == 0) {
    return TERLAN_NDARRAY_DLPACK_DTYPE;
  }
  if (tensor->ndim < 0 || tensor->ndim > TERLAN_NDARRAY_DLPACK_MAX_RANK ||
      (tensor->ndim > 0 && tensor->shape == NULL) ||
      tensor->byte_offset != 0) {
    return TERLAN_NDARRAY_DLPACK_LAYOUT;
  }
  for (index = 0; index < tensor->ndim; ++index) {
    if (tensor->shape[index] < 0) {
      return TERLAN_NDARRAY_NEGATIVE_DIMENSION;
    }
    if (tensor->shape[index] == 0) {
      numel = 0;
    } else if (numel != 0) {
      if (numel > INT64_MAX / tensor->shape[index]) {
        return TERLAN_NDARRAY_SIZE_OVERFLOW;
      }
      numel *= tensor->shape[index];
    }
  }
  if (tensor->strides != NULL) {
    for (index = tensor->ndim; index > 0; --index) {
      if (tensor->strides[index - 1] != expected_stride) {
        return TERLAN_NDARRAY_DLPACK_LAYOUT;
      }
      if (tensor->shape[index - 1] == 0) {
        expected_stride = 0;
      } else if (expected_stride != 0) {
        if (expected_stride > INT64_MAX / tensor->shape[index - 1]) {
          return TERLAN_NDARRAY_SIZE_OVERFLOW;
        }
        expected_stride *= tensor->shape[index - 1];
      }
    }
  }
  element_size = dtype_size(*out_dtype);
  if ((uint64_t)numel > SIZE_MAX / element_size ||
      (numel > 0 && tensor->data == NULL)) {
    return numel > 0 && tensor->data == NULL ? TERLAN_NDARRAY_DLPACK_LAYOUT
                                              : TERLAN_NDARRAY_SIZE_OVERFLOW;
  }
  if (numel > 0 && (uintptr_t)tensor->data % element_size != 0) {
    return TERLAN_NDARRAY_DLPACK_LAYOUT;
  }
  if (*out_dtype == TERLAN_NDARRAY_DTYPE_BOOL) {
    const uint8_t *values = tensor->data;
    for (index = 0; index < numel; ++index) {
      if (values[index] > 1) {
        return TERLAN_NDARRAY_INVALID_BOOL;
      }
    }
  }
  *out_numel = numel;
  return TERLAN_NDARRAY_OK;
}

TerlanNdArrayStatus terlan_ndarray_v1_from_dlpack(
    DLManagedTensorVersioned *managed, TerlanNdArrayHandle *out_array) {
  TerlanNdArrayStatus status = TERLAN_NDARRAY_INVALID_ARGUMENT;
  int32_t dtype = 0;
  int64_t numel = 0;

  if (out_array != NULL) {
    *out_array = NULL;
  }
  if (managed == NULL || out_array == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  if (managed->deleter == NULL) {
    return TERLAN_NDARRAY_DLPACK_DELETER;
  }
  if (managed->version.major != DLPACK_MAJOR_VERSION) {
    status = TERLAN_NDARRAY_DLPACK_VERSION;
  } else if ((managed->flags &
              ~(DLPACK_FLAG_BITMASK_READ_ONLY |
                DLPACK_FLAG_BITMASK_IS_COPIED)) != 0) {
    status = TERLAN_NDARRAY_DLPACK_LAYOUT;
  } else {
    status = validate_import(&managed->dl_tensor, &dtype, &numel);
    if (status == TERLAN_NDARRAY_OK) {
      status = create_array(
          managed->dl_tensor.data, numel, managed->dl_tensor.shape,
          managed->dl_tensor.ndim, dtype_size(dtype), dtype, out_array);
    }
  }
  managed->deleter(managed);
  return status;
}

int64_t terlan_ndarray_v1_dlpack_deleter_count(void) {
  return atomic_load_explicit(&dlpack_deleter_count, memory_order_relaxed);
}

static void write_u16_le(uint8_t *output, uint16_t value) {
  output[0] = (uint8_t)value;
  output[1] = (uint8_t)(value >> 8);
}

static void write_u32_le(uint8_t *output, uint32_t value) {
  size_t index;
  for (index = 0; index < 4; ++index) {
    output[index] = (uint8_t)(value >> (index * 8));
  }
}

static void write_u64_le(uint8_t *output, uint64_t value) {
  size_t index;
  for (index = 0; index < 8; ++index) {
    output[index] = (uint8_t)(value >> (index * 8));
  }
}

static uint16_t read_u16_le(const uint8_t *input) {
  return (uint16_t)input[0] | (uint16_t)((uint16_t)input[1] << 8);
}

static uint32_t read_u32_le(const uint8_t *input) {
  uint32_t value = 0;
  size_t index;
  for (index = 0; index < 4; ++index) {
    value |= (uint32_t)input[index] << (index * 8);
  }
  return value;
}

static uint64_t read_u64_le(const uint8_t *input) {
  uint64_t value = 0;
  size_t index;
  for (index = 0; index < 8; ++index) {
    value |= (uint64_t)input[index] << (index * 8);
  }
  return value;
}

static uint8_t native_endian_code(void) {
  const uint16_t value = 1;
  return *(const uint8_t *)&value == 1 ? 1 : 2;
}

TerlanNdArrayStatus terlan_ndarray_v1_export_tensor_packet(
    TerlanNdArrayHandle array, const char *consumer, uint8_t **out_packet,
    size_t *out_length) {
  DLManagedTensorVersioned *managed = NULL;
  DLTensor *tensor;
  uint8_t *packet = NULL;
  size_t metadata_size;
  size_t total_size;
  size_t index;
  size_t element_size;
  size_t data_size;
  TerlanNdArrayStatus status;

  if (consumer == NULL || consumer[0] == '\0' || out_packet == NULL ||
      out_length == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_packet = NULL;
  *out_length = 0;
  status = terlan_ndarray_v1_to_dlpack(array, &managed);
  if (status != TERLAN_NDARRAY_OK) {
    return status;
  }
  tensor = &managed->dl_tensor;
  element_size = (size_t)tensor->dtype.bits / 8;
  data_size = (size_t)array->numel * element_size;
  if ((size_t)tensor->ndim >
      (SIZE_MAX - TERLAN_TENSOR_PACKET_HEADER_SIZE) / 16) {
    status = TERLAN_NDARRAY_SIZE_OVERFLOW;
    goto cleanup;
  }
  metadata_size =
      TERLAN_TENSOR_PACKET_HEADER_SIZE + (size_t)tensor->ndim * 16;
  if (data_size > SIZE_MAX - metadata_size) {
    status = TERLAN_NDARRAY_SIZE_OVERFLOW;
    goto cleanup;
  }
  total_size = metadata_size + data_size;
  packet = checked_malloc(total_size);
  if (packet == NULL) {
    status = TERLAN_NDARRAY_ALLOCATION_FAILED;
    goto cleanup;
  }
  memcpy(packet, "TNXP", 4);
  packet[4] = 1;
  packet[5] = native_endian_code();
  packet[6] = tensor->dtype.code;
  packet[7] = tensor->dtype.bits;
  write_u16_le(packet + 8, tensor->dtype.lanes);
  packet[10] = (uint8_t)tensor->device.device_type;
  packet[11] = (uint8_t)tensor->device.device_id;
  write_u32_le(packet + 12, (uint32_t)tensor->ndim);
  for (index = 0; index < (size_t)tensor->ndim; ++index) {
    write_u64_le(packet + TERLAN_TENSOR_PACKET_HEADER_SIZE + index * 8,
                 (uint64_t)tensor->shape[index]);
    write_u64_le(packet + TERLAN_TENSOR_PACKET_HEADER_SIZE +
                     (size_t)tensor->ndim * 8 + index * 8,
                 (uint64_t)tensor->strides[index]);
  }
  if (data_size > 0) {
    memcpy(packet + metadata_size, tensor->data, data_size);
  }
  *out_packet = packet;
  *out_length = total_size;
  status = TERLAN_NDARRAY_OK;

cleanup:
  managed->deleter(managed);
  return status;
}

static void delete_import_packet(DLManagedTensorVersioned *managed) {
  delete_managed_tensor(managed);
}

TerlanNdArrayStatus terlan_ndarray_v1_import_tensor_packet(
    const uint8_t *packet, int64_t packet_length,
    TerlanNdArrayHandle *out_array) {
  TerlanNdArrayDLPackManager *manager = NULL;
  DLTensor *tensor;
  uint32_t rank;
  size_t metadata_size;
  size_t data_size;
  size_t index;
  size_t element_size;
  TerlanNdArrayStatus status = TERLAN_NDARRAY_DLPACK_LAYOUT;

  if (out_array == NULL || packet_length < TERLAN_TENSOR_PACKET_HEADER_SIZE ||
      packet == NULL) {
    return TERLAN_NDARRAY_INVALID_ARGUMENT;
  }
  *out_array = NULL;
  if (memcmp(packet, "TNXP", 4) != 0 || packet[4] != 1 ||
      packet[5] != native_endian_code()) {
    return TERLAN_NDARRAY_DLPACK_VERSION;
  }
  rank = read_u32_le(packet + 12);
  if (rank > TERLAN_NDARRAY_DLPACK_MAX_RANK) {
    return TERLAN_NDARRAY_DLPACK_LAYOUT;
  }
  metadata_size = TERLAN_TENSOR_PACKET_HEADER_SIZE + (size_t)rank * 16;
  if ((uint64_t)packet_length < metadata_size) {
    return TERLAN_NDARRAY_DLPACK_LAYOUT;
  }
  data_size = (size_t)packet_length - metadata_size;
  manager = checked_calloc(1, sizeof(*manager));
  if (manager == NULL) {
    return TERLAN_NDARRAY_ALLOCATION_FAILED;
  }
  tensor = &manager->tensor.dl_tensor;
  if (rank > 0) {
    tensor->shape = checked_malloc((size_t)rank * sizeof(int64_t));
    tensor->strides = checked_malloc((size_t)rank * sizeof(int64_t));
    if (tensor->shape == NULL || tensor->strides == NULL) {
      destroy_export(manager);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    for (index = 0; index < rank; ++index) {
      tensor->shape[index] =
          (int64_t)read_u64_le(packet + TERLAN_TENSOR_PACKET_HEADER_SIZE +
                               index * 8);
      tensor->strides[index] = (int64_t)read_u64_le(
          packet + TERLAN_TENSOR_PACKET_HEADER_SIZE + (size_t)rank * 8 +
          index * 8);
    }
  }
  if (data_size > 0) {
    tensor->data = checked_malloc(data_size);
    if (tensor->data == NULL) {
      destroy_export(manager);
      return TERLAN_NDARRAY_ALLOCATION_FAILED;
    }
    memcpy(tensor->data, packet + metadata_size, data_size);
  }
  tensor->device.device_type = (DLDeviceType)packet[10];
  tensor->device.device_id = packet[11];
  tensor->ndim = (int32_t)rank;
  tensor->dtype.code = packet[6];
  tensor->dtype.bits = packet[7];
  tensor->dtype.lanes = read_u16_le(packet + 8);
  tensor->byte_offset = 0;
  element_size = (size_t)tensor->dtype.bits / 8;
  if (element_size == 0 || data_size % element_size != 0) {
    destroy_export(manager);
    return TERLAN_NDARRAY_DLPACK_DTYPE;
  }
  manager->tensor.version.major = DLPACK_MAJOR_VERSION;
  manager->tensor.version.minor = DLPACK_MINOR_VERSION;
  manager->tensor.manager_ctx = manager;
  manager->tensor.deleter = delete_import_packet;
  {
    int32_t validated_dtype = 0;
    int64_t validated_numel = 0;
    size_t expected_data_size;
    status = validate_import(tensor, &validated_dtype, &validated_numel);
    if (status != TERLAN_NDARRAY_OK) {
      manager->tensor.deleter(&manager->tensor);
      return status;
    }
    element_size = dtype_size(validated_dtype);
    expected_data_size = (size_t)validated_numel * element_size;
    if (data_size != expected_data_size) {
      manager->tensor.deleter(&manager->tensor);
      return TERLAN_NDARRAY_DLPACK_LAYOUT;
    }
  }
  status = terlan_ndarray_v1_from_dlpack(&manager->tensor, out_array);
  return status;
}

void terlan_ndarray_v1_delete_tensor_packet(uint8_t *packet) { free(packet); }
