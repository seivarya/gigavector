#ifndef GIGAVECTOR_GV_TURBOQUANT_H
#define GIGAVECTOR_GV_TURBOQUANT_H

#include <stddef.h>
#include <stdint.h>

#include "search/distance.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GV_TURBOQUANT_ROTATION_AUTO = 0,
    GV_TURBOQUANT_ROTATION_FHWT = 1,
    GV_TURBOQUANT_ROTATION_QR = 2
} GV_TurboQuantRotation;

typedef struct {
    uint8_t bits;              /**< Total bits per scalar (2-16). Polar uses bits-1 when QJL enabled. */
    size_t projections;        /**< QJL sketch dimension (0 disables QJL). */
    uint64_t seed;             /**< Deterministic seed for rotation and QJL matrices. */
    int use_qjl;               /**< 1 = PolarQuant (bits-1) + QJL residual; 0 = PolarQuant only. */
    GV_TurboQuantRotation rotation;
} GV_TurboQuantConfig;

typedef struct GV_TurboQuantCode GV_TurboQuantCode;
typedef struct GV_TurboQuantizer GV_TurboQuantizer;

typedef struct {
    float *rotated_query;
    float *qjl_projected;
    size_t projections;
} GV_TurboQuantQuery;

GV_TurboQuantizer *turboquant_create(size_t dimension, const GV_TurboQuantConfig *config);
void turboquant_destroy(GV_TurboQuantizer *quantizer);

GV_TurboQuantCode *turboquant_encode(const GV_TurboQuantizer *quantizer, const float *vector);
void turboquant_code_destroy(GV_TurboQuantCode *code);

int turboquant_prepare_query(const GV_TurboQuantizer *quantizer, const float *query,
                             GV_TurboQuantQuery *out);
void turboquant_query_destroy(GV_TurboQuantQuery *query);

float turboquant_inner_product(const GV_TurboQuantizer *quantizer, const GV_TurboQuantCode *code,
                               const GV_TurboQuantQuery *query);
float turboquant_l2_squared(const GV_TurboQuantizer *quantizer, const GV_TurboQuantCode *code,
                            const GV_TurboQuantQuery *query);
float turboquant_distance(const GV_TurboQuantizer *quantizer, const GV_TurboQuantCode *code,
                          const float *query, GV_DistanceType distance_type);
float turboquant_distance_prepared(const GV_TurboQuantizer *quantizer,
                                   const GV_TurboQuantCode *code,
                                   const GV_TurboQuantQuery *query,
                                   GV_DistanceType distance_type);

size_t turboquant_dimension(const GV_TurboQuantizer *quantizer);

#ifdef __cplusplus
}
#endif

#endif
