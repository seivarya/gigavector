/**
 * @file fuzz_wal_apply.c
 * @brief libFuzzer harness for wal_apply_record_buffer (WAL record parser).
 *
 * Build: make fuzz-wal-apply  (requires clang)
 * Run:   build/fuzz/fuzz_wal_apply tests/fuzz/corpus/wal -runs=10000
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "storage/wal.h"

#define FUZZ_MAX_INPUT (256 * 1024)
#define FUZZ_DIM 8

static int noop_insert(void *ctx, const float *data, size_t dimension,
                       const char *const *metadata_keys,
                       const char *const *metadata_values,
                       size_t metadata_count) {
    (void)ctx;
    (void)data;
    (void)dimension;
    (void)metadata_keys;
    (void)metadata_values;
    (void)metadata_count;
    return 0;
}

static int noop_delete(void *ctx, size_t vector_index) {
    (void)ctx;
    (void)vector_index;
    return 0;
}

static int noop_update(void *ctx, size_t vector_index, const float *data, size_t dimension,
                       const char *const *metadata_keys, const char *const *metadata_values,
                       size_t metadata_count) {
    (void)ctx;
    (void)vector_index;
    (void)data;
    (void)dimension;
    (void)metadata_keys;
    (void)metadata_values;
    (void)metadata_count;
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > FUZZ_MAX_INPUT) return 0;

    (void)wal_apply_record_buffer(data, size, 1, FUZZ_DIM,
                                  noop_insert, noop_delete, noop_update, NULL, NULL);
    (void)wal_apply_record_buffer(data, size, 0, FUZZ_DIM,
                                  noop_insert, noop_delete, noop_update, NULL, NULL);
    return 0;
}

#ifdef GV_FUZZ_STANDALONE
#include <stdio.h>
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input-file>\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    uint8_t buf[FUZZ_MAX_INPUT];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    return LLVMFuzzerTestOneInput(buf, n);
}
#endif
