#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "storage/turboquant.h"

#define ASSERT(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return -1; } } while(0)

#define DIM 8

static void fill_vector(float *data, size_t dim, float base) {
    for (size_t i = 0; i < dim; i++) {
        data[i] = sinf(base + (float)i);
    }
}

static int test_turboquant_create_destroy(void) {
    GV_TurboQuantConfig config = {
        .bits = 8,
        .projections = 2,
        .seed = 42,
        .use_qjl = 1,
        .rotation = GV_TURBOQUANT_ROTATION_FHWT
    };

    GV_TurboQuantizer *q = turboquant_create(DIM, &config);
    ASSERT(q != NULL, "turboquant_create returned NULL");
    ASSERT(turboquant_dimension(q) == DIM, "dimension mismatch");
    turboquant_destroy(q);

    q = turboquant_create(DIM, NULL);
    ASSERT(q != NULL, "turboquant_create with NULL config failed");
    turboquant_destroy(q);

    return 0;
}

static int test_turboquant_encode_distance(void) {
    float data[DIM];
    fill_vector(data, DIM, 0.0f);

    GV_TurboQuantConfig config = {
        .bits = 8,
        .projections = 2,
        .seed = 42,
        .use_qjl = 1,
        .rotation = GV_TURBOQUANT_ROTATION_FHWT
    };

    GV_TurboQuantizer *q = turboquant_create(DIM, &config);
    ASSERT(q != NULL, "create failed");

    GV_TurboQuantCode *code = turboquant_encode(q, data);
    ASSERT(code != NULL, "encode failed");

    float query[DIM];
    fill_vector(query, DIM, 0.5f);

    float dist = turboquant_distance(q, code, query, GV_DISTANCE_EUCLIDEAN);
    ASSERT(dist >= 0.0f, "distance should be non-negative");

    GV_TurboQuantQuery prepared;
    ASSERT(turboquant_prepare_query(q, query, &prepared) == 0, "prepare_query failed");
    float dist2 = turboquant_distance_prepared(q, code, &prepared, GV_DISTANCE_EUCLIDEAN);
    ASSERT(fabsf(dist - dist2) < 1e-5f, "prepared distance mismatch");
    turboquant_query_destroy(&prepared);

    turboquant_code_destroy(code);
    turboquant_destroy(q);
    return 0;
}

static int test_turboquant_odd_dimension_rejected(void) {
    GV_TurboQuantizer *q = turboquant_create(7, NULL);
    ASSERT(q == NULL, "odd dimension should be rejected");
    return 0;
}

int main(void) {
    int rc = 0;
    rc |= test_turboquant_create_destroy();
    rc |= test_turboquant_encode_distance();
    rc |= test_turboquant_odd_dimension_rejected();
    return rc;
}
