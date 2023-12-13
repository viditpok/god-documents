#include "mmu.h"
#include "pagesim.h"
#include "swapops.h"
#include "stats.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

/**
 * --------------------------------- PROBLEM 6 --------------------------------------
 * Checkout PDF section 7 for this problem
 * 
 * Page fault handler.
 * 
 * When the CPU encounters an invalid address mapping in a page table, it invokes the 
 * OS via this handler. Your job is to put a mapping in place so that the translation 
 * can succeed.
 * 
 * @param addr virtual address in the page that needs to be mapped into main memory.
 * 
 * HINTS:
 *      - You will need to use the global variable current_process when
 *      altering the frame table entry.
 *      - Use swap_exists() and swap_read() to update the data in the 
 *      frame as it is mapped in.
 * ----------------------------------------------------------------------------------
 */
void page_fault(vaddr_t addr) {
   // TODO: Get a new frame, then correctly update the page table and frame table
   vpn_t vpn = vaddr_vpn(addr);
   pte_t * entryP = (pte_t * )(mem + PTBR * PAGE_SIZE) + vpn;

   pfn_t pfn = free_frame();
   entryP->pfn = pfn;
   entryP->valid = 1;
   entryP->dirty = 0;

   fte_t * ft = frame_table + pfn;
   ft->protected = 0;
   ft->mapped = 1;
   ft->referenced = 1;
   ft->process = current_process;
   ft->vpn = vpn;

   uint8_t * temp = mem + pfn * PAGE_SIZE;
   if (swap_exists(entryP)) {
      swap_read(entryP, temp);
   } else {
      (void) memset(temp, 0, PAGE_SIZE);
   }


   stats.page_faults++;
}

#pragma GCC diagnostic pop
