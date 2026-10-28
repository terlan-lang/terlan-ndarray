#define _POSIX_C_SOURCE 200809L

#include "../bindings/terlan_ndarray_internal.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint64_t elapsed_nanoseconds(struct timespec start,
                                    struct timespec finish) {
  const uint64_t seconds = (uint64_t)(finish.tv_sec - start.tv_sec);
  const int64_t nanoseconds = finish.tv_nsec - start.tv_nsec;
  return seconds * UINT64_C(1000000000) + (uint64_t)nanoseconds;
}

static int execute_product(TerlanNdArrayHandle lhs, TerlanNdArrayHandle rhs,
                           int64_t size, int sample) {
  TerlanNdArrayHandle result = NULL;
  double *values = NULL;
  size_t length = 0;
  struct timespec start;
  struct timespec finish;
  uint64_t duration;

  if (clock_gettime(CLOCK_MONOTONIC, &start) != 0 ||
      terlan_ndarray_v1_matmul(lhs, rhs, &result) != TERLAN_NDARRAY_OK ||
      clock_gettime(CLOCK_MONOTONIC, &finish) != 0) {
    return 1;
  }
  duration = elapsed_nanoseconds(start, finish);
  if (terlan_ndarray_v1_to_floats(result, &values, &length) !=
          TERLAN_NDARRAY_OK ||
      length != (size_t)(size * size)) {
    (void)terlan_ndarray_v1_delete(result);
    return 1;
  }
  printf("size=%" PRId64 " sample=%d elapsed_ns=%" PRIu64
         " checksum=%.17g\n",
         size, sample, duration,
         length == 0 ? 0.0 : values[0] + values[length - 1]);
  terlan_ndarray_v1_delete_float_values(values);
  return terlan_ndarray_v1_delete(result) == TERLAN_NDARRAY_OK ? 0 : 1;
}

int main(int argc, char **argv) {
  char *end = NULL;
  long parsed;
  int64_t size;
  int64_t shape[2];
  size_t element_count;
  double *lhs_values;
  double *rhs_values;
  TerlanNdArrayHandle lhs = NULL;
  TerlanNdArrayHandle rhs = NULL;
  TerlanNdArrayHandle warmup = NULL;
  int sample;

  if (argc != 2) {
    return 2;
  }
  parsed = strtol(argv[1], &end, 10);
  if (*argv[1] == '\0' || *end != '\0' || parsed <= 0 || parsed > 1024) {
    return 2;
  }
  size = (int64_t)parsed;
  shape[0] = size;
  shape[1] = size;
  element_count = (size_t)size * (size_t)size;
  lhs_values = malloc(element_count * sizeof(*lhs_values));
  rhs_values = malloc(element_count * sizeof(*rhs_values));
  if (lhs_values == NULL || rhs_values == NULL) {
    free(lhs_values);
    free(rhs_values);
    return 1;
  }
  for (size_t index = 0; index < element_count; ++index) {
    lhs_values[index] = (double)((index % 17) + 1) / 17.0;
    rhs_values[index] = (double)((index % 13) + 1) / 13.0;
  }
  if (terlan_ndarray_v1_from_floats(lhs_values, (int64_t)element_count, shape,
                                    2, &lhs) != TERLAN_NDARRAY_OK ||
      terlan_ndarray_v1_from_floats(rhs_values, (int64_t)element_count, shape,
                                    2, &rhs) != TERLAN_NDARRAY_OK) {
    free(lhs_values);
    free(rhs_values);
    (void)terlan_ndarray_v1_delete(lhs);
    (void)terlan_ndarray_v1_delete(rhs);
    return 1;
  }
  free(lhs_values);
  free(rhs_values);
  if (terlan_ndarray_v1_matmul(lhs, rhs, &warmup) != TERLAN_NDARRAY_OK ||
      terlan_ndarray_v1_delete(warmup) != TERLAN_NDARRAY_OK) {
    return 1;
  }
  for (sample = 0; sample < 3; ++sample) {
    if (execute_product(lhs, rhs, size, sample) != 0) {
      return 1;
    }
  }
  if (terlan_ndarray_v1_delete(lhs) != TERLAN_NDARRAY_OK ||
      terlan_ndarray_v1_delete(rhs) != TERLAN_NDARRAY_OK ||
      terlan_ndarray_v1_live_count() != 0) {
    return 1;
  }
  return 0;
}
