#include <stdio.h>
#include <string.h>

#include "swap.h"
#include "util.h"

static uint64_t SWAP_ID = 1;

/**
 * Create a swap entry with a unique ID.
 * 
 * @return newly created entry
 */
swap_entry_t *create_entry(void)
{
    swap_entry_t *new_entry = calloc(1, sizeof(swap_entry_t));
    if (!new_entry) {
        panic("could not allocate swap entry");
    }
    new_entry->id = SWAP_ID++;
    return new_entry;
}

/**
 * Enqueue a new swap entry to the swap queue.
 * 
 * @param queue pointer to the swap queue
 * @param new_entry pointer to the entry to be enqueued
 */
void swap_queue_enqueue(swap_queue_t *queue, swap_entry_t* new_entry)
{
    new_entry->next = NULL;
    if (queue->head == NULL) {
        queue->head = queue->tail = new_entry;
    } else {
        queue->tail->next = new_entry;
        queue->tail = new_entry;
    }
    queue->size++;
    if (queue->size > queue->size_max) {
        queue->size_max = queue->size;
    }
}

/**
 * Dequeue and free the specified swap entry from the swap queue
 * 
 * @param queue pointer to the swap queue
 * @param sid id of the swap entry to be dequeued
 */
void swap_queue_dequeue(swap_queue_t *queue, swap_id_t sid)
{
    swap_entry_t *curr = queue->head;
    swap_entry_t *prev = NULL;

    while (curr) {
        if (curr->id == sid) {
            if (prev) {
                prev->next = curr->next;
            } else {
                queue->head = curr->next;
            }
            if (curr == queue->tail) {
                queue->tail = prev;
            }
            break;
        }
        prev = curr;
        curr = curr->next;
    }
    queue->size--;
    curr->next = NULL;
    free(curr);
}

/**
 * Find and return the specified swap entry from the swap queue
 * 
 * @param queue pointer to the swap queue
 * @param sid id of the swap entry to be found
 * 
 * @return pointer to the found swap_entry
 */
swap_entry_t *swap_queue_find(swap_queue_t *queue, swap_id_t sid)
{
    swap_entry_t *curr = queue->head;

    while (curr) {
        if (curr->id == sid) {
            return curr;
        }
        curr = curr->next;
    }
    return NULL;
}
