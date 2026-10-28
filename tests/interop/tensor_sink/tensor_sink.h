#ifndef TERLAN_TENSOR_SINK_H
#define TERLAN_TENSOR_SINK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t terlan_tensor_sink_v1_consume(const uint8_t *packet,
                                      int64_t packet_length,
                                      int64_t *out_sum);
int32_t terlan_tensor_sink_v1_consume_pair(const uint8_t *first_packet,
                                           int64_t first_packet_length,
                                           const uint8_t *second_packet,
                                           int64_t second_packet_length,
                                           int64_t *out_sum);
int32_t terlan_tensor_sink_v1_reject(const uint8_t *packet,
                                     int64_t packet_length,
                                     int64_t *out_sum);
int32_t terlan_tensor_sink_v1_crash(const uint8_t *packet,
                                    int64_t packet_length,
                                    int64_t *out_sum);

#ifdef __cplusplus
}
#endif

#endif
