/**
 * Name: Vidit D. Pokharna
 * GTID: 903772087
 */

/*  PART 2: A CS-2200 C implementation of the arraylist data structure.
    Implement an array list.
    The methods that are required are all described in the header file. Description for the methods can be found there.

    Hint 1: Review documentation/ man page for malloc, calloc, and realloc.
    Hint 2: Review how an arraylist works.
    Hint 3: You can use GDB if your implentation causes segmentation faults.
*/

#include "arraylist.h"

arraylist_t *create_arraylist(uint capacity) {
    arraylist_t *new_arraylist = (arraylist_t * ) malloc(sizeof(arraylist_t));
    if (new_arraylist == NULL) {
        return NULL;
    }
    new_arraylist->capacity = capacity;
    new_arraylist->size = 0;
    new_arraylist->backing_array = (char **) malloc(sizeof(char *) * capacity);
    if (new_arraylist->backing_array == NULL) {
        free(new_arraylist);
        return NULL;
    }
    return new_arraylist;
}

void add_at_index(arraylist_t *arraylist, char *data, int index) {
    if ((index < 0) || (index > arraylist->size) || data == NULL) {
        return;
    }
    if (arraylist->capacity == arraylist->size) {
        resize(arraylist);
    }
    for (int i = arraylist->size; i > index; i--) {
        arraylist->backing_array[i] = arraylist->backing_array[i - 1];
    }
    arraylist->backing_array[index] = data;
    arraylist->size++;
}

void append(arraylist_t *arraylist, char *data) {
    if (data == NULL || arraylist == NULL) {
        return;
    }
    add_at_index(arraylist, data, arraylist->size);
}

char *remove_from_index(arraylist_t *arraylist, int index) {
    if (index < 0 || index >= arraylist->size || arraylist == NULL) {
        return NULL;
    }
    char *data = 0;
    data = arraylist->backing_array[index];
    if (index != arraylist->size - 1) {
        for (int i = index; i < arraylist->size - 1; i++) {
            arraylist->backing_array[i] = arraylist->backing_array[i + 1];
        }
    }
    arraylist->backing_array[arraylist->size-1] = 0;
    arraylist->size--;
    return data;
}

void resize(arraylist_t *arraylist) {
    if (arraylist == NULL) {
        return;
    }
    char **resizedArraylist = (char **) realloc(arraylist->backing_array, sizeof(char *) * arraylist->capacity * 2);
    if (resizedArraylist == NULL) {
        return;
    }
    arraylist->capacity = arraylist->capacity * 2;
    arraylist->backing_array = resizedArraylist;
}

void destroy(arraylist_t *arraylist) {
    if (arraylist == NULL) {
        return;
    }
    free(arraylist->backing_array);
}