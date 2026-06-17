#ifndef GV_DST_HARNESS_H
#define GV_DST_HARNESS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * Seeded deterministic RNG for DST (Deterministic Simulation Testing).
 * Same seed => same pseudo-random sequence across platforms.
 */
typedef struct {
    uint64_t state;
} GV_DstRng;

static inline GV_DstRng gv_dst_rng_seed(uint64_t seed) {
    GV_DstRng rng;
    rng.state = seed ? seed : 0x475656454354ULL; /* "GVVECT" */
    return rng;
}

static inline uint64_t gv_dst_rng_next(GV_DstRng *rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 2685821657736338717ULL;
}

static inline uint32_t gv_dst_rng_u32(GV_DstRng *rng, uint32_t max_exclusive) {
    if (max_exclusive <= 1) return 0;
    return (uint32_t)(gv_dst_rng_next(rng) % max_exclusive);
}

static inline float gv_dst_rng_float(GV_DstRng *rng) {
    return (float)((gv_dst_rng_next(rng) >> 11) & 0xFFFFF) / (float)(1 << 20);
}

static inline uint64_t gv_dst_seed_from_env(void) {
    const char *env = getenv("GV_DST_SEED");
    if (!env || !*env) return 0x4756473235ULL; /* "GVGV25" default */
    return (uint64_t)strtoull(env, NULL, 0);
}

static inline size_t gv_dst_iters_from_env(size_t default_iters) {
    const char *env = getenv("GV_DST_ITERS");
    if (!env || !*env) return default_iters;
    unsigned long v = strtoul(env, NULL, 0);
    return v > 0 ? (size_t)v : default_iters;
}

#endif
