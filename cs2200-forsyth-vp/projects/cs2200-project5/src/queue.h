#pragma once

#include <stdlib.h>

/*
 * A struct for a basic queue.
 *
 * The queue struct holds the list of elements as well as
 * the current number of elements.
 */
typedef struct queue {
    int size;
    void **elements;
} queue_t;

/* Initializes the struct with an empty queue and size 0 */
void queue_init(queue_t *queue);

/* Adds the specified element to the queue and increments size by 1 */
void queue_add(queue_t *queue, void *element);

/* Removes first element in queue and decrements size by 1 */
void *queue_extract(queue_t *queue);

/* Returns current number of elements in the queue */
int queue_size(queue_t *queue);

/* Releases any memory and resources held internally by the queue */
void queue_release(queue_t *queue);
