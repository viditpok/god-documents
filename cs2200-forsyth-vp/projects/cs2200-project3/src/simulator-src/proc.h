#pragma once

#include "swap.h"
#include "types.h"
#include "pagesim.h"
#include "va_splitting.h"

/**
 * Process functions.
 *
 * These will be completed by you in the corresponding files, proc.c,
 * page_fault.c, and page_replacement.c.
 */

void proc_init(pcb_t *proc);
void context_switch(pcb_t *proc);
void proc_cleanup(pcb_t *proc);
