#include "tensor_sink.h"

#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define TENSOR_PACKET_HEADER_SIZE 16
#define TENSOR_PACKET_RANK 2
#define TENSOR_PACKET_METADATA_SIZE                                           \
  (TENSOR_PACKET_HEADER_SIZE + TENSOR_PACKET_RANK * 16)
#define TENSOR_PACKET_ELEMENT_COUNT 6
#define TENSOR_PACKET_SIZE                                                    \
  (TENSOR_PACKET_METADATA_SIZE + TENSOR_PACKET_ELEMENT_COUNT * sizeof(int64_t))

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

static int64_t read_i64_le(const uint8_t *input) {
  uint64_t value = 0;
  size_t index;
  for (index = 0; index < 8; ++index) {
    value |= (uint64_t)input[index] << (index * 8);
  }
  return (int64_t)value;
}

static uint8_t native_endian_code(void) {
  const uint16_t value = 1;
  return *(const uint8_t *)&value == 1 ? 1 : 2;
}

static int32_t validate_and_sum(const uint8_t *packet, int64_t packet_length,
                                int64_t *out_sum) {
  int64_t sum = 0;
  size_t index;
  if (packet == NULL || out_sum == NULL || packet_length != TENSOR_PACKET_SIZE) {
    return 200;
  }
  *out_sum = 0;
  if (memcmp(packet, "TNXP", 4) != 0 || packet[4] != 1 ||
      packet[5] != native_endian_code() || packet[6] != 0 || packet[7] != 64 ||
      read_u16_le(packet + 8) != 1 || packet[10] != 1 || packet[11] != 0 ||
      read_u32_le(packet + 12) != TENSOR_PACKET_RANK ||
      read_i64_le(packet + 16) != 2 || read_i64_le(packet + 24) != 3 ||
      read_i64_le(packet + 32) != 3 || read_i64_le(packet + 40) != 1) {
    return 200;
  }
  for (index = 0; index < TENSOR_PACKET_ELEMENT_COUNT; ++index) {
    const int64_t value =
        read_i64_le(packet + TENSOR_PACKET_METADATA_SIZE + index * 8);
    if ((value > 0 && sum > INT64_MAX - value) ||
        (value < 0 && sum < INT64_MIN - value)) {
      return 202;
    }
    sum += value;
  }
  *out_sum = sum;
  return 0;
}

int32_t terlan_tensor_sink_v1_consume(const uint8_t *packet,
                                      int64_t packet_length,
                                      int64_t *out_sum) {
  return validate_and_sum(packet, packet_length, out_sum);
}

int32_t terlan_tensor_sink_v1_consume_pair(const uint8_t *first_packet,
                                           int64_t first_packet_length,
                                           const uint8_t *second_packet,
                                           int64_t second_packet_length,
                                           int64_t *out_sum) {
  int64_t first_sum;
  int64_t second_sum;
  int32_t status =
      validate_and_sum(first_packet, first_packet_length, &first_sum);
  if (status != 0) {
    return status;
  }
  status = validate_and_sum(second_packet, second_packet_length, &second_sum);
  if (status != 0) {
    return status;
  }
  if (first_sum > INT64_MAX - second_sum) {
    return 202;
  }
  *out_sum = first_sum + second_sum;
  return 0;
}

int32_t terlan_tensor_sink_v1_reject(const uint8_t *packet,
                                     int64_t packet_length,
                                     int64_t *out_sum) {
  const int32_t status = validate_and_sum(packet, packet_length, out_sum);
  return status == 0 ? 201 : status;
}

int32_t terlan_tensor_sink_v1_crash(const uint8_t *packet,
                                    int64_t packet_length,
                                    int64_t *out_sum) {
  const int32_t status = validate_and_sum(packet, packet_length, out_sum);
  if (status != 0) {
    return status;
  }
  _Exit(86);
}
