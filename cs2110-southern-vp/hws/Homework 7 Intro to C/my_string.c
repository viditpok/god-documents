/**
 * @file my_string.c
 * @author Vidit Pokharna
 * @collaborators NAMES OF PEOPLE THAT YOU COLLABORATED WITH HERE
 * @brief Your implementation of these famous 3 string.h library functions!
 *
 * NOTE: NO ARRAY NOTATION IS ALLOWED IN THIS FILE
 *
 * @date 2023-03-xx
 */

#include <stddef.h>
#include "my_string.h"
/**
 * @brief Calculate the length of a string
 *
 * @param s a constant C string
 * @return size_t the number of characters in the passed in string
 */
size_t my_strlen(const char *s)
{
    int sum = 0;
    while(*s != '\0') {
        sum++;
        s++;
    }
    return sum;
}

/**
 * @brief Compare two strings
 *
 * @param s1 First string to be compared
 * @param s2 Second string to be compared
 * @param n First (at most) n bytes to be compared
 * @return int "less than, equal to, or greater than zero if s1 (or the first n
 * bytes thereof) is found, respectively, to be less than, to match, or be
 * greater than s2"
 */
int my_strncmp(const char *s1, const char *s2, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (*s1 != *s2 || *s1 == '\0' || *s2 == '\0') {
            return (*s1 - *s2);
        }
        s1++;
        s2++;
    }
    return 0;
}

/**
 * @brief Copy a string
 *
 * @param dest The destination buffer
 * @param src The source to copy from
 * @param n maximum number of bytes to copy
 * @return char* a pointer same as dest
 */
char *my_strncpy(char *dest, const char *src, size_t n)
{
    char *p = dest;
    while (n > 0 && *src != '\0') {
        *p++ = *src++;
        --n;
    }
    while (n > 0) {
        *p++ = '\0';
        --n;
    }
    return dest;
}

/**
 * @brief Concatenates two strings and stores the result
 * in the destination string
 *
 * @param dest The destination string
 * @param src The source string
 * @param n The maximum number of bytes from src to concatenate
 * @return char* a pointer same as dest
 */
char *my_strncat(char *dest, const char *src, size_t n)
{
    char *dest_end = dest;
    while (*dest_end != '\0') {
        dest_end++;
    }
    size_t i = 0;
    while (i < n && *src != '\0') {
        *dest_end = *src;
        dest_end++;
        src++;
        i++;
    }
    *dest_end = '\0';
    return dest;
}

/**
 * @brief Copies the character c into the first n
 * bytes of memory starting at *str
 *
 * @param str The pointer to the block of memory to fill
 * @param c The character to fill in memory
 * @param n The number of bytes of memory to fill
 * @return char* a pointer same as str
 */
void *my_memset(void *str, int c, size_t n)
{
    char *p = str;
    char value = c;
    for (size_t i = 0; i < n; i++) {
        *(p + i) = value;
    }
    return str;
}

/**
 * @brief Finds the first instance of c in str
 * and removes it from str in place
 *
 * @param str The pointer to the string
 * @param c The character we are looking to delete
 */
void remove_first_instance(char *str, char c){
    char *p = str;
    int found = 0;
    while (*p != '\0') {
        if (*p == c) {
            found = 1;
            break;
        }
        p++;
    }
    if (found) {
        char *q = p;
        while (*q != '\0') {
            *q = *(q + 1);
            q++;
        }
    }
}

/**
 * @brief Finds the first instance of c in str
 * and replaces it with the contents of replaceStr
 *
 * @param str The pointer to the string
 * @param c The character we are looking to delete
 * @param replaceStr The pointer to the string we are replacing c with
 */
void replace_character_with_string(char *str, char c, char *replaceStr) {
   int replaceStrLen = my_strlen(replaceStr);
   char *curr = str;
   while (*curr != 0) {
       if (*curr == c) {
           if (*replaceStr == 0) {
               remove_first_instance(str, c);
               break;
           }
           int strLen = 0;
           char *tempLen = curr + 1;
           while (*tempLen != 0) {
               strLen++;
               tempLen++;
           }
           char *locInStr = curr + strLen;
           char *locNew = curr + replaceStrLen + strLen - 1;
           while (strLen > 0) {
               *locNew =  *(locInStr);
               locInStr--;
               locNew--;
               strLen--;
           }
           while (replaceStrLen > 0) {
               *curr = *replaceStr;
               curr++;
               replaceStr++;
               replaceStrLen--;
           }
           break;
       }
       curr++;
   }
}


/**
 * @brief Remove the first character of str (ie. str[0]) IN ONE LINE OF CODE.
 * No loops allowed. Assume non-empty string
 * @param str A pointer to a pointer of the string
 */
void remove_first_character(char **str) {
    *str = *str + 1;
}