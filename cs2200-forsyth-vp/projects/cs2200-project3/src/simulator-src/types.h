#pragma once

#include <inttypes.h> /* For uintXX_t types */

/* Virtual addresses are stored in a 32-bit integer. */
typedef uint32_t vaddr_t;

/* Physical addresses are stored in a 32-bit integer. */
typedef uint32_t paddr_t;

/* Virtual page numbers can be up to 16 bits. For pedantic reasons. */
typedef uint16_t vpn_t;

/* Physical frame numbers can be up to 16 bits. For pedantic reasons. */
typedef uint16_t pfn_t;

/* This machine is byte addressed, so an unsigned char will suffice. */
typedef unsigned char word_t;

/* This is used to store the address of the swap entry on disk*/
typedef uint64_t swap_id_t;

/* Used in the simulator to tell how long a process runs */
typedef uint32_t timestamp_t;
