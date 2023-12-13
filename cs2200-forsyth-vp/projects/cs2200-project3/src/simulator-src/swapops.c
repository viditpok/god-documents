#include <stdlib.h>

#include "swapops.h"
#include "util.h"

swap_queue_t swap_queue;

/**
 * Reads bytes from a page table entry's swap entry into physical
 * memory. Copies PAGE_SIZE bytes from swap to memory starting at dst.
 *
 * @param entry a pointer to the page table entry
 * @param dst the destination to which bytes read from swap should be
 * copied. This should be a pointer to the start of the relevant frame
 * in your mem[] array.
 */
void swap_read(pte_t *pte, void *dst) {

    swap_entry_t *info = swap_queue_find(&swap_queue, pte->sid);
    if (!info) {
        panic("Attempted to read an invalid swap entry.\nHINT: How do you check if a swap entry exists, and if it does not, what should you put in memory instead?");
    }
    memcpy(dst, info->page_data, PAGE_SIZE);
}

/**
 * Writes bytes from physical memory into a swap space corresponding
 * to the supplied page table entry. Copies PAGE_SIZE bytes from
 * memory starting at src into swap.
 *
 * If no swap entry is currently allocated for the page, a new one
 * will be allocated. This entry must later be freed with swap_free to
 * avoid a memory leak. The page table entry's swap field is updated
 * automatically.
 *
 * @param entry a pointer to the page table entry
 * @param src the source address from which bytes should be copied
 * into swap. This should be a pointer to the start of the relevant
 * frame in your mem[] array.
 */
void swap_write(pte_t *pte, void *src) {

    swap_entry_t *info = swap_queue_find(&swap_queue, pte->sid);
    if (!info) {
        info = create_entry();
        swap_queue_enqueue(&swap_queue, info);
        pte->sid = info->id;
    }
    memcpy(info->page_data, src, PAGE_SIZE);
}

/**
 * Frees the swap entry associated with the given page table entry.
 *
 * The space held by the swap entry is de-allocated and the swap field
 * in the page table is automatically cleared.
 *
 * @param entry a pointer to the page table entry
 */
void swap_free(pte_t *pte) {
    swap_id_t swp_entry = pte->sid;
    if (!swap_queue_find(&swap_queue, swp_entry)) {
        panic("Attempted to free an invalid swap entry!");
    }
    swap_queue_dequeue(&swap_queue, pte->sid);
    pte->sid = 0;
}