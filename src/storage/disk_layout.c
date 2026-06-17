#include "storage/disk_layout.h"

size_t gv_disk_default_sector_size(void)
{
    return GV_DISK_SECTOR_SIZE_DEFAULT;
}

size_t gv_disk_normalize_sector_size(size_t sector_size)
{
    return sector_size ? sector_size : GV_DISK_SECTOR_SIZE_DEFAULT;
}
