#pragma once

#include "mmu.h"
#include "proc.h"
#include "pagesim.h"
//#include "paging.h"
#include "swap.h"
#include "types.h"

extern swap_queue_t swap_queue;

/**
 * Determines if the given page table entry has a swap entry.
 *
 * @param entry a pointer to the page table entry to check
 */
static inline int swap_exists(pte_t *entry) {
    return entry->sid != 0;
}

void swap_read(pte_t *entry, void *dst);
void swap_write(pte_t *entry, void *src);
void swap_free(pte_t *entry);

