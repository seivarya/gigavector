#include "storage/database.h"
#include "storage/wal.h"
#include "admin/replication.h"

GV_Database *gv_db_open(const char *filepath, size_t dimension, GV_IndexType index_type) {
    return db_open(filepath, dimension, index_type);
}

GV_Database *gv_db_open_with_hnsw_config(const char *filepath, size_t dimension,
                                        GV_IndexType index_type, const GV_HNSWConfig *hnsw_config) {
    return db_open_with_hnsw_config(filepath, dimension, index_type, hnsw_config);
}

GV_Database *gv_db_open_with_ivfpq_config(const char *filepath, size_t dimension,
                                         GV_IndexType index_type, const GV_IVFPQConfig *ivfpq_config) {
    return db_open_with_ivfpq_config(filepath, dimension, index_type, ivfpq_config);
}

void gv_db_close(GV_Database *db) {
    db_close(db);
}

int gv_db_add_vector(GV_Database *db, const float *data, size_t dimension) {
    return db_add_vector(db, data, dimension);
}

int gv_db_add_vector_with_metadata(GV_Database *db, const float *data, size_t dimension,
                                  const char *metadata_key, const char *metadata_value) {
    return db_add_vector_with_metadata(db, data, dimension, metadata_key, metadata_value);
}

int gv_db_add_vector_with_rich_metadata(GV_Database *db, const float *data, size_t dimension,
                                      const char *const *metadata_keys, const char *const *metadata_values,
                                      size_t metadata_count) {
    return db_add_vector_with_rich_metadata(db, data, dimension, metadata_keys, metadata_values, metadata_count);
}

int gv_db_save(const GV_Database *db, const char *filepath) {
    return db_save(db, filepath);
}

int gv_db_search(const GV_Database *db, const float *query_data, size_t k,
                 GV_SearchResult *results, GV_DistanceType distance_type) {
    return db_search(db, query_data, k, results, distance_type);
}

int gv_db_search_filtered(const GV_Database *db, const float *query_data, size_t k,
                          GV_SearchResult *results, GV_DistanceType distance_type,
                          const char *filter_key, const char *filter_value) {
    return db_search_filtered(db, query_data, k, results, distance_type, filter_key, filter_value);
}

int gv_db_search_batch(const GV_Database *db, const float *queries, size_t qcount, size_t k,
                       GV_SearchResult *results, GV_DistanceType distance_type) {
    return db_search_batch(db, queries, qcount, k, results, distance_type);
}

int gv_db_ivfpq_train(GV_Database *db, const float *data, size_t count, size_t dimension) {
    return db_ivfpq_train(db, data, count, dimension);
}

void gv_replication_config_init(GV_ReplicationConfig *config) {
    replication_config_init(config);
}

GV_ReplicationManager *gv_replication_create(GV_Database *db, const GV_ReplicationConfig *config) {
    return replication_create(db, config);
}

void gv_replication_destroy(GV_ReplicationManager *mgr) {
    replication_destroy(mgr);
}

int gv_replication_start(GV_ReplicationManager *mgr) {
    return replication_start(mgr);
}

int gv_replication_stop(GV_ReplicationManager *mgr) {
    return replication_stop(mgr);
}

int gv_replication_add_follower(GV_ReplicationManager *mgr, const char *node_id, const char *address) {
    return replication_add_follower(mgr, node_id, address);
}

int gv_replication_sync_commit(GV_ReplicationManager *mgr, uint32_t timeout_ms) {
    return replication_sync_commit(mgr, timeout_ms);
}

int gv_replication_leader_append_wal(GV_ReplicationManager *mgr, uint64_t entry_delta, uint64_t byte_delta) {
    return replication_leader_append_wal(mgr, entry_delta, byte_delta);
}

int gv_wal_truncate(GV_WAL *wal) {
    return wal_truncate(wal);
}

