// 2200 Project 3 
// Spring 2022
// Paging simulation

#include <stdio.h>
#include <getopt.h>

#include "pagesim.h"
#include "mmu.h"
#include "proc.h"
#include "swap.h"
#include "stats.h"
#include "swapops.h"

uint8_t *mem;                       // start of our simulated memory
pfn_t PTBR;                         // page table base register
pcb_t *current_process;             // currently running process
uint8_t replacement = 0;            // page replacment policy
timestamp_t step = 0;               // current number of steps in the program

uint8_t check_corruption = 0;       // the corruption checker flag is set  

/* Internal array of running processes (we only expose current_process to the user) */
static pcb_t *procs;

/* Constants used in parsing the trace file */
static const char *START = "START";
static const char *STOP = "STOP";

/* static methods to run the simulation */
static FILE* read_args(int argc, char **argv);

static void sim_cmd(const char* cmd);
static void sim_start_proc(uint32_t pid);
static void sim_stop_proc(uint32_t pid);
static void sim_mem_access(uint32_t pid, char rw, uint32_t address, uint8_t data);

static void print_help_and_exit(void);
static void check_validity(int checks);

int main(int argc, char **argv) 
{
    /* Allocate some memory! */
    if (!(mem = calloc(1, MEM_SIZE)))
        exit(1);

    /* Allocate procs */
    if (!(procs = calloc(MAX_PID, sizeof(pcb_t))))
        exit(1);

    /* Read command line options */
    FILE* fin = read_args(argc, argv);

    /* Start the simulation by calling the student implemented system_init()*/
    system_init();
    
    if (check_corruption) 
        check_validity(0);

    /* read trace line to buf */
    char buf[120];
    while ((fgets(buf, sizeof(buf), fin))) 
    {
        sim_cmd(buf);       // perform the trace line memeory operation
        step++;             // Increment the timestamp
    }
    fclose(fin);

    /* Cleanup and print statistics */
    free(mem);
    free(procs);
    compute_stats();

    printf("Total Accesses     : %" PRIu64 "\n", stats.accesses);
    printf("Page Faults        : %" PRIu64 "\n", stats.page_faults);
    printf("Writes to disk     : %" PRIu64 "\n", stats.writebacks);
    printf("Average Memory Access Time: %f\n", stats.amat);
    printf("Max Swap Size      : %" PRIu64 " KB\n", (((uint64_t) swap_queue.size_max) * PAGE_SIZE) >> 10);

    if (swap_queue.size > 0)
        printf("Swap Not Freed     : %" PRIu64 " KB\n", (((uint64_t) swap_queue.size) * PAGE_SIZE) >> 10);
}

/* read command line arguments of the simulation */
FILE* read_args(int argc, char **argv)
{
    FILE *fin = 0;
    int opt;
    while (-1 != (opt = getopt(argc, argv, "i:hscr:"))) {
        switch (opt) {
        case 'i':   // specify the trace file
            fin = fopen(optarg, "r");

            if (!fin) {
                perror("Unable to open trace file");
                exit(1);
            }
            break;
        case 's':   // can input the trace lines
            fin = stdin;
            printf("Input trace lines.\n");
            break;
        case 'c':   // turn corruption checker on
            check_corruption = 1;
            printf("-> Note: Strict memory corruption checking is enabled.\n");
            break;
        case 'r':   // specify the replacement algorithm 
            if (strcmp(optarg, "random") == 0) {
                replacement = RANDOM;
            } else if (strcmp(optarg, "fifo") == 0) {
                replacement = FIFO;
            } else if (strcmp(optarg, "clocksweep") == 0) {
                replacement = CLOCKSWEEP;
            } else {
                fprintf(stderr, "Unknown replacement algorithm: %s", optarg);
                exit(1);
            }
            break;
        case 'h':   // print usage message
        default:
            print_help_and_exit();
        }
    }

    /* errors in input */
    if (!fin) {
        fprintf(stderr, "ERROR: You must specify a trace filename or stdin.\n");
        print_help_and_exit();
    }
    if (!replacement) {
        fprintf(stderr, "ERROR: You must select a replacement algorithm using -r.\n");
        print_help_and_exit();
    }

    return fin;
}

/*
 * The sim_cmd function is run for each line/command in the trace file
 * There are three types of commands:
 *     Start Process:  START <PID>
 *     Stop Process:   STOP <PID>
 *     Memory Access:  <PID> <r/w> <Address> <Value>
 */
void sim_cmd(const char* cmd)
{
    char rw;
    uint8_t data;
    uint32_t address;
    uint32_t pid;

    if (!strncmp(cmd, START, 5)) {  // Start Process Command
        int ret = sscanf(cmd+6, "%" PRIu32 "\n", &pid);  // Get the PID from the command

        if (ret == 1) {
            sim_start_proc(pid);
        }
        else {
            printf("Unable to parse trace file: Invalid START command encountered\n");
            exit(1);
        }
    }
    else if (!strncmp(cmd, STOP, 4)) {  // Stop Process Command
        /* Start scanning from the pid digits */
        int ret = sscanf((cmd+5), "%" PRIu32 "\n", &pid);
        if (ret == 1) {
            sim_stop_proc(pid);
        }
        else {
            printf("Unable to parse trace file: Invalid STOP command encountered\n");
            exit(1);
        }
    }
    else {  // Memory Access Command
        int ret = sscanf(cmd, "%u %c %x %hhu\n", &pid, &rw, &address, &data);

        if (ret == 4) {
            sim_mem_access(pid, rw, address, data);
        }
        else {
            printf("Unable to parse trace file: Invalid memory access command encountered\n");
            exit(1);
        }
    }
}

/* start a new process in the simulation */
void sim_start_proc(uint32_t pid)
{
    pcb_t *new_proc = &procs[pid];
    new_proc->pid = pid;
    new_proc->state = PROC_RUNNING;
    proc_init(new_proc);

    printf("%8u: PID %u started\n", step, pid);
    if (check_corruption)
        check_validity(1);
}

/* stop a process and cleanup its memory */
void sim_stop_proc(uint32_t pid)
{
    /* clean up the memory footprints of procs[pid] student implmented proc_cleanup() */
    proc_cleanup(&procs[pid]);
    procs[pid].saved_ptbr = 0;
    procs[pid].state = PROC_STOPPED;

    printf("%8u: PID %u stopped\n", step, pid);
    if (check_corruption)
        check_validity(1);
}

/* perform a memory operation for the specified process */
void sim_mem_access(uint32_t pid, char rw, uint32_t address, uint8_t data) {
    /* If a process is currently running, and it's not this one, do a context switch */
    if (!current_process || current_process->pid != pid) {
        context_switch(&procs[pid]);
        current_process = &procs[pid];
    }

    /* Get the new data value from the student implemented mem_access() */
    uint8_t new_data = mem_access(address, rw, data);

    /* Print data for trace verification */
    if (rw == 'r')
        printf("%8u: %3u  r  0x%05x -> %02hhx\n", step, pid, address, new_data);
    else 
        printf("%8u: %3u  w  0x%05x <- %02hhx\n", step, pid, address, data);

    if (check_corruption)
        check_validity(1);
}

/* checks the correctness of the memory operations */
void check_validity(int checks) {
    uint32_t pid, vpn, pfn, running_procs;
    uint8_t protected_frames_accounted_for[NUM_FRAMES];
    uint8_t mapped_frames_accounted_for[NUM_FRAMES];
    for (pfn = 0; pfn < NUM_FRAMES; pfn++) {
        protected_frames_accounted_for[pfn] = 0;
        mapped_frames_accounted_for[pfn] = 0;
    }

    /* Validate frame table is set up correctly */
    if ((void *)frame_table != (void *) mem) {
        panic("Frame table should begin at the first frame in memory");
    }

    if (!frame_table[0].protected) {
        panic("Frame 0 should be marked as protected");
    }
    protected_frames_accounted_for[0] = 1;

    if (checks < 1) return;

    /* Validate the PTBRs are correct */
    running_procs = 0;
    for (pid = 0; pid < MAX_PID; pid++) {
        if (procs[pid].state == PROC_RUNNING) {
            running_procs++;

            /* Validate that PTBR points to a correct physical frame number */
            pfn_t found_ptbr = procs[pid].saved_ptbr;
            if (found_ptbr <= 0 || found_ptbr > NUM_FRAMES)  {
                panic("PTBR of running process cannot be zero or >= the number of frames in the system");
            }

            /* Validate that page table page is marked as protected */
            if (!frame_table[found_ptbr].protected) {
                panic("Frames corresponding to the page tables of running processes must be marked as protected");
            }
            protected_frames_accounted_for[found_ptbr] = 1;
        }
    }

    /* Check for any protected frames that should not be protected */
    for (pfn = 0; pfn < NUM_FRAMES; pfn++) {
        if (frame_table[pfn].protected && !protected_frames_accounted_for[pfn]) {
            panic("Found frame marked as protected that should not be protected");
        }
    }

    /* Validate the page table entries are correct */
    for (pid = 0; pid < MAX_PID; pid++) {
        if (procs[pid].state == PROC_RUNNING) {
            pfn_t found_ptbr = procs[pid].saved_ptbr;

            /* Scan the entire page table, make sure frame table is
               consistent with any valid pages */
            pte_t *pgtable = (pte_t *)(mem + (found_ptbr * PAGE_SIZE));
            for (vpn = 0; vpn < NUM_PAGES; vpn++) {
                /* Check basic sanity of boolean flags */
                if (pgtable[vpn].valid != 0 && pgtable[vpn].valid != 1) {
                    panic("Page table entry valid bit should either be zero or one");
                }

                if (pgtable[vpn].dirty != 0 && pgtable[vpn].dirty != 1) {
                    panic("Page table entry dirty bit should either be zero or one");
                }

                /* If valid, check sanity of pfn */
                if (pgtable[vpn].valid) {
                    pfn_t found_pfn = pgtable[vpn].pfn;

                    /* Check basic ranges */
                    if (found_pfn <= 0 || found_pfn > NUM_FRAMES - 1)  {
                        panic("PFN of page table entry cannot be zero or >= the number of frames in the system");
                    }

                    if (protected_frames_accounted_for[found_pfn]) {
                        panic("Page table entry should not map to a protected frame");
                    }

                    if (mapped_frames_accounted_for[found_pfn]) {
                        panic("Duplicate PFN found in page table");
                    }

                    if (frame_table[found_pfn].process < procs
                        || frame_table[found_pfn].process >= procs + MAX_PID) {
                        panic("Mapped frame table entry contains invalid process pointer");
                    }

                    /* Check that frame table agrees with page table */
                    if (!frame_table[found_pfn].mapped
                        || !(frame_table[found_pfn].process->pid == pid)
                        || !(frame_table[found_pfn].vpn == vpn)) {
                        panic("Frame table is inconsistent with page table entry");
                    }
                    mapped_frames_accounted_for[found_pfn] = 1;
                }

                /* Check the validity of swap entry */
                if (pgtable[vpn].sid && !swap_queue_find(&swap_queue, pgtable[vpn].sid)) {
                    panic("Page table entry points to swap entry that does not exist");
                }
            }
        }
    }

    /* Check that all frames that are mapped are accounted for */
    for (pfn = 0; pfn < NUM_FRAMES; pfn++){
        if (!frame_table[pfn].protected && frame_table[pfn].mapped && !(mapped_frames_accounted_for[pfn])) {
            panic("Found frame table entry marked as mapped with no corresponding page table entry");
        }
    }
}

// print the utilization 
void print_help_and_exit() {
    printf("./vm-sim [OPTIONS] -i traces/file.trace -r<replacement algorithm>\n");
    printf("  -i\t\tReads the trace from the specified path\n");
    printf("  -s\t\tReads the trace from standard input\n");
    printf("  -r\t\tSelect the replacement algorithm (either 'random' or 'fifo' or 'clocksweep')\n");
    printf("  -c\t\tEnables strict memory corruption checking\n");
    printf("    \t\t(automatically checks a variety of conditions that can cause bugs)\n");
    printf("  -h\t\tThis helpful output\n");
    exit(0);
}