#ifndef TERLAN_NDARRAY_INTERNAL_H
#define TERLAN_NDARRAY_INTERNAL_H

#include "include/terlan_ndarray.h"

#include <stdint.h>

typedef enum TerlanNdArrayState {
  TERLAN_NDARRAY_STATE_LIVE = 1,
  TERLAN_NDARRAY_STATE_DISPOSED = 2
} TerlanNdArrayState;

struct TerlanNdArray {
  int32_t dtype;
  int32_t device_type;
  int64_t rank;
  int64_t numel;
  int64_t byte_count;
  int64_t *shape;
  int64_t *strides;
  void *data;
  TerlanNdArrayState state;
};

void terlan_ndarray_test_fail_allocation_after(int64_t successful_allocations);
void terlan_ndarray_test_reset_allocation_failures(void);

#endif
