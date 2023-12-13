// READ THE HEADER FILE FOR MORE DOCUMENTATION
#include "tl04.h"

/**
 * 2110’s Ed Discussion page becomes extremely busy sometimes. After some consultation with the 3510 TAs,
 * the 2110 TAs determined the most efficient method of answering posts was following the first-in, last-out
 * method. 
 *
 * In this timed lab, you will implement a stack that will be used to manage Ed posts. To 
 * implement this stack, you will use a array of pointers. That means each array element contains
 * a pointer to an object of type post_t.
 * In order to help you manage the data structure, we have provided two global pointers. There is capacity
 * for the size of the backing array and numPosts for the number of items held in the stack. You will be given
 * responsibility to change these variables as needed.
 * 
 * Note that dynamic memory allocation is required for this implementation. It’s unknown if you will have to
 * increase the size of the backing array at compilation.
 * 
 * For this stack, you shall be writing three functions that will aid TAs with keeping track of students on the
 * stack: push_stack, pop_stack, and destroy_stack
 * 
 */


/**
 * \brief Variables used to track the stack.
 *
 * The two integers (capacity and numPosts) represent the current 
 * maximum capacity of posts Ed Discussion can hold and the current number 
 * of posts respectively.
 * 
 * The pointer (stack_arr) points to the bottom of the stack.
 *
 * Initially, capacity is set to 5, numPosts is set to 0, and stack_arr is
 * pointing the first open spot at the bottom of the stack. The stack is initially
 * empty so stack_arr is originally pointing to garbage data.
 *
 * \property extern int capacity
 * \property extern int numPosts
 * \property extern struct post_t **stack_arr
 */
int capacity = 5;
int numPosts = 0;
struct post_t **stack_arr;


/**
 * \brief Provided function to initialize the stack
*/
void initialize_stack(void) {
    stack_arr = (struct post_t **) malloc(sizeof(struct post_t*) * capacity);
}

/**
 * \brief Add posts to the top of the stack
 *
 * This function will be called by client code to add a post to the top of
 * the stack. The caller will supply the question and category of the post to add.
 *
 * This function should allocate a [post_t] on the heap, and deep-copy all the data.
 * In particular, any pointers in the [post_t] will require their own dedicated memory allocation.
 * Make sure that all members of the [post_t] are set!
 *
 * If the stack is full, double the capacity of the stack and then add the new post to the stack.
 * 
 * Finally, insert the post onto the stack with the help of the [stack_arr] pointer.
 * Refer back to the PDF/diagram for specific details about how the stack works, 
 * and consider any edge cases.
 * 
 * This function should return `SUCCESS` if the post was added successfully.
 * If it fails, it should return `FAILURE` and leave the list unchanged. It
 * should fail if and only if:
 * * `malloc` or `realloc` fails,
 * * the post's question is `NULL`, or
 * * the post's question is an empty string.
 *
 * \param[in] question The question for the post
 * \param[in] category The category for the post
 * \return Whether the post was successfully added
 */
int stack_push(const char *question, enum category_t category) {
    if (stack_arr == NULL || question == NULL || strcmp(question, "") == 0) {
        return FAILURE;
    }

    struct post_t* insert = (struct post_t*) malloc(sizeof(struct post_t));
    if (insert == NULL) {
        return FAILURE;
    }

    insert->category = category;
    insert->question = malloc(strlen(question) + 1);
    if (insert->question == NULL) {
        free(insert);
        return FAILURE;
    }
    strcpy(insert->question, question);

    if (numPosts == capacity) {
        struct post_t** stack_arr_temp = (struct post_t **) malloc(sizeof(struct post_t*) * capacity * 2);
        if (stack_arr_temp == NULL) {
            free(insert->question);
            free(insert);
            return FAILURE;
        }
        for (int i = 0; i < numPosts; i++) {
            stack_arr_temp[i] = stack_arr[i];
        }
        free(stack_arr);
        stack_arr = stack_arr_temp;
        capacity *= 2;
    }

    stack_arr[numPosts] = insert;
    numPosts++;

    return SUCCESS;
}

/**
 * \brief Pop a question from the stack
 *
 * This function will be called by client code to remove a post from the
 * top of the stack. It will return whether a post was removed successfully,
 * and the post removed in that case.
 *
 * The way this function returns the post is using the data out technique.
 * This is to get around the limitation that functions may only have one return
 * value. As such, the caller will pass in a pointer where the post
 * should be stored. Then this function will store the returned post at that
 * pointer. Independently, it returns whether it succeeded via the normal path.
 * 
 * Finally, set the pointer of the post being popped to 'NULL' and update numPosts.
 * 
 * Refer back to the PDF/diagram for specific details about how to the stack 
 * works, and consider any edge cases.
 *
 * If this function succeeds, it should return `SUCCESS` and modify `*data` to
 * be pointing to the removed post. If it fails, it should return `FAILURE`
 * and leave both the stack and `*data` unchanged. It should fail if and only if:
 * * [data] is `NULL`, or
 * * the stack is empty.
 * 
 * Remember to free any unused data. In this case, the pointer to the question string is still
 * used in *data so do not free it!
 *
 * \param[out] data Where to put the removed post
 * \return Whether a post was successfully removed
 */
int stack_pop(struct post_t *data) {
    if (stack_arr == NULL || data == NULL || numPosts == 0) {
        return FAILURE;
    }

    struct post_t* remove = stack_arr[numPosts - 1];
    *data = *remove;

    free(remove);

    stack_arr[numPosts - 1] = NULL;
    numPosts--;

    return SUCCESS;
}

/**
 * \brief Destroy the whole stack
 *
 * This function will be called by client code to free the whole stack. This involves
 * freeing all elements associated with the stack: the stack's backing array, every post
 * associated with the stack, and the data associated with each post. Finally, you will set
 * numPosts to 0.
 * 
 * When this function succeeds, it should return `SUCCESS`.
 */
int destroy_stack(void) {
    for (int i = 0; i < numPosts; i++) {
        free(stack_arr[i]->question);
        free(stack_arr[i]);
    }
    free(stack_arr);
    numPosts = 0;
    return SUCCESS;
}