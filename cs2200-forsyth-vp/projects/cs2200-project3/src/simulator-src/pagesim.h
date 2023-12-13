#pragma once

#include "types.h"

/* Maximum number of possible processes */
#define MAX_PID 800

/* Process's current state */
#define PROC_RUNNING 1
#define PROC_STOPPED 0

/* Length of physical addresses */
#define PADDR_LEN 20
/* Length of virtual addresses */
#define VADDR_LEN 24
/* Length of offset */
#define OFFSET_LEN 14
/* Size of virtual page found from the offset */
#define PAGE_SIZE (1 << OFFSET_LEN)
/* Total size of physical memory */
#define MEM_SIZE (1 << PADDR_LEN)
/* Toal number of pages in virtual address space */
#define NUM_PAGES (1 << (VADDR_LEN - OFFSET_LEN))
/* Toal number of pages in physical address space */
#define NUM_FRAMES (1 << (PADDR_LEN - OFFSET_LEN))

/* Different replacement strategies */
#define RANDOM 1
#define FIFO 2
#define CLOCKSWEEP 3


/*
 * A process control block (PCB).
 *
 * PCBs hold the necessary state to facilitate switching between different
 * processes running on the system at any point in time.
 */
typedef struct process {
    uint32_t pid;
    uint8_t state;
    pfn_t saved_ptbr;
} pcb_t;


extern uint8_t *mem;            // Simulated physical memory 
extern pfn_t PTBR;              // The page table base register
extern pcb_t *current_process;  // The currently running process

extern uint8_t replacement;     // Selected replacement strategy
