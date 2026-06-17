#ifndef GIGAVECTOR_GV_DISK_PAGE_CACHE_H
#define GIGAVECTOR_GV_DISK_PAGE_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GV_DiskPageCache GV_DiskPageCache;

typedef struct {
    size_t cache_hits;
    size_t cache_misses;
    size_t cached_entries;
    size_t used_bytes;
    size_t max_bytes;
} GV_DiskPageCacheStats;

GV_DiskPageCache *gv_disk_page_cache_create(size_t max_bytes);
void gv_disk_page_cache_destroy(GV_DiskPageCache *cache);

void gv_disk_page_cache_set_max_bytes(GV_DiskPageCache *cache, size_t max_bytes);
void gv_disk_page_cache_get_stats(const GV_DiskPageCache *cache, GV_DiskPageCacheStats *out);

/**
 * @brief Lookup cached bytes by key. Returns pointer valid until eviction.
 */
const uint8_t *gv_disk_page_cache_lookup(GV_DiskPageCache *cache, const char *key, size_t *len_out);

/**
 * @brief Insert or replace an entry (copies @p data).
 */
int gv_disk_page_cache_insert(GV_DiskPageCache *cache, const char *key,
                              const uint8_t *data, size_t len);

void gv_disk_page_cache_remove(GV_DiskPageCache *cache, const char *key);

#ifdef __cplusplus
}
#endif

#endif
