#pragma once

#include "pagesim.h"
#include "types.h"

/**
 * an entry into swap queue that holds individual pages saved to the swap space 
 */
typedef struct swap_entry {
    uint64_t id;                    // entry's id. Used to find it in queue
    uint8_t  page_data[PAGE_SIZE];  // page data saved to swap space

    struct swap_entry *next;        // pointer to next swap entry
} swap_entry_t;

/**
 * a queue of swap entries that simulate the swap space on disk 
 */
typedef struct _swap_queue_t {
    swap_entry_t *head;             // start of current swap space
    swap_entry_t *tail;             // end of current swap space
    uint64_t size;                  // size of current swap space
    uint64_t size_max;              // max size the swap space can be
} swap_queue_t;

swap_entry_t *create_entry(void);
void swap_queue_enqueue(swap_queue_t *queue, swap_entry_t* info);
void swap_queue_dequeue(swap_queue_t *queue, uint64_t sid);
swap_entry_t *swap_queue_find(swap_queue_t *queue, swap_id_t sid);
