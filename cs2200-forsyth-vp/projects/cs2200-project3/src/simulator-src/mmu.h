#pragma once

#include "swap.h"
#include "types.h"
#include "pagesim.h"
#include "va_splitting.h"

/**
 * An entry in the frame table.
 *
 * The frame table is a global table that keeps information about each frame 
 * in physical memory.
 */
typedef struct ft_entry {
    uint8_t protected;          /* set if the frame holds a page that should be
                                immune from eviction */
    uint8_t mapped;             /* set if the frame is mapped. */
    uint8_t referenced;         /* set if the entry has been recently accessed. */         
    pcb_t *process;             /* A pointer to the owning process's PCB */
    vpn_t vpn;                  /* The VPN mapped by the process using this frame. */
} fte_t;

/**
 * An entry in the page table.
 * 
 * Note that the VPN is not stored in the entry - it's the index into 
 * the page table!
 */
typedef struct ptable_entry {
    uint8_t valid;              /* set if the entry is mapped to a valid frame. */
    uint8_t dirty;              /* set if the entry is modified while in main 
                                memory */
    pfn_t pfn;                  /* The physical frame number (PFN) this entry
                                maps to. */
    swap_id_t sid;          /* The swap entry mapped to this page. Use this
                                to read to/write from the page to disk using
                                swap_read() and swap_write() */
} pte_t;


/* A convenient global reference to the frame table, which you will
   set up in mmu.c */
extern fte_t *frame_table;

/**
 * Mmu functions
 * 
 * These will be completed by you in mmu.c, page_fault.c, and page_replacment.c
 */
void system_init(void);
uint8_t mem_access(vaddr_t address, char write, uint8_t data);

pfn_t free_frame(void);
void page_fault(vaddr_t address);

