;;=============================================================
;;  CS 2110 - Spring 2023
;;  Homework 6 - Factorial
;;=============================================================
;;  Name: Vidit Pokharnac
;;============================================================

;;  In this file, you must implement the 'MULTIPLY' and 'FACTORIAL' subroutines.

;;  Little reminder from your friendly neighborhood 2110 TA staff: don't run
;;  this directly by pressing 'Run' in complx, since there is nothing put at
;;  address x3000. Instead, call the subroutine by doing the following steps:
;;      * 'Debug' -> 'Simulate Subroutine Call'
;;      * Call the subroutine at the 'MULTIPLY' or 'FACTORIAL' labels
;;      * Add the [a, b] or [n] params separated by a comma (,) 
;;        (e.g. 3, 5 for 'MULTIPLY' or 6 for 'FACTORIAL')
;;      * Proceed to run, step, add breakpoints, etc.
;;      * Remember R6 should point at the return value after a subroutine
;;        returns. So if you run the program and then go to 
;;        'View' -> 'Goto Address' -> 'R6 Value', you should find your result
;;        from the subroutine there (e.g. 3 * 5 = 15 or 6! = 720)

;;  If you would like to setup a replay string (trace an autograder error),
;;  go to 'Test' -> 'Setup Replay String' -> paste the string (everything
;;  between the apostrophes (')) excluding the initial " b' ". If you are 
;;  using the Docker container, you may need to use the clipboard (found
;;  on the left panel) first to transfer your local copied string over.

.orig x3000
    ;; You do not need to write anything here
    HALT

;;  MULTIPLY Pseudocode (see PDF for explanation and examples)   
;;  
;;  MULTIPLY(int a, int b) {
;;      int ret = 0;
;;      while (b > 0) {
;;          ret += a;
;;          b--;
;;      }
;;      return ret;
;;  }

MULTIPLY ;; Do not change this label! Treat this as like the name of the function in a function header
    ; Stack set up

	; Space for RV
	ADD R6, R6, -1

	; Push Return Address
	ADD R6, R6, -1
	STR R7, R6, 0

	; Save old frame pointer	
	ADD R6, R6, -1
	STR R5, R6, 0

	;Make space for first local variable
	ADD R6, R6, -1

	; Stack and frame pointer point to first local
	; fp = sp
	ADD R5, R6, 0   ;; R5 = R6

    ADD     R6, R6, #-5	    ;; Make space for old REGs (assuming only 1 LV)
    STR     R0, R6, #0      ;; Store old R0
    STR     R1, R6, #1      ;; Store old R1
    STR     R2, R6, #2      ;; Store old R2
    STR     R3, R6, #3      ;; Store old R3
    STR     R4, R6, #4      ;; Store old R4

	;------Subroutine Stuff------

    ; R0 = ret = 0
	AND R0, R0, #0

	; Load b into R1 -> R1 = b
	LDR R1, R5, 5
    ADD R1, R1, 0

    ; Load a into R2 -> R2 = a
    LDR R2, R5, 4
    ADD R2, R2, 0

WHILE
    ADD R1, R1, #0;

	; if b <= 0, return ret
	BRnz BASECASE

	; b >= 1

    ; ret += a -> R0 = R0 + R2
    ; b-- -> R1 = R1 - 1
    ADD R0, R0, R2
    ADD R1, R1, -1
    
    BR WHILE

BASECASE
	STR R0, R5, 3

TEARDOWN
    ;; Tear down the stack
    LDR     R0, R6, #0      ;; Restore old R0
    LDR     R1, R6, #1      ;; Restore old R1
    LDR     R2, R6, #2      ;; Restore old R2
    LDR     R3, R6, #3      ;; Restore old R3
    LDR     R4, R6, #4      ;; Restore old R4
    ADD     R6, R5, #0      ;; Pop off restored registers and any local variables (LV)
    LDR     R5, R6, #1      ;; Restore old frame pointer (FP)
    LDR     R7, R6, #2      ;; Restore return address (RA)
    ADD     R6, R6, #3      ;; Pop off LV1, old FP, and RA
    RET

FACTORIAL ;; Do not change this label! Treat this as like the name of the function in a function header
    
    ; Stack set up

	; Space for RV
	ADD R6, R6, -1

	; Push Return Address
	ADD R6, R6, -1
	STR R7, R6, 0

	; Save old frame pointer	
	ADD R6, R6, -1
	STR R5, R6, 0

	;Make space for first local variable
	ADD R6, R6, -1

	; Stack and frame pointer point to first local
	; fp = sp
	ADD R5, R6, 0   ;; R5 = R6

    ADD     R6, R6, -5	    ;; Make space for old REGs (assuming only 1 LV)
    STR     R0, R6, #0      ;; Store old R0
    STR     R1, R6, #1      ;; Store old R1
    STR     R2, R6, #2      ;; Store old R2
    STR     R3, R6, #3      ;; Store old R3
    STR     R4, R6, #4      ;; Store old R4

	;------Subroutine Stuff------

	; R0 = ret = 1
	AND R0, R0, #0
    ADD R0, R0, #1

    ; Load n into R1 -> R1 = n
	LDR R1, R5, 4
    ADD R1, R1, 0

    ; R3 = x = 0
	AND R3, R0, #0
    ADD R3, R3, 2

FOR
    ; if (x - n > 0), return ret
    NOT R4, R1
    ADD R4, R4, #1
    ADD R4, R4, R3

    BRp BASECASE1
	
	; Make recursive call
	; We are now the caller

	; Push arguments
	ADD R6, R6, -1
	STR R3, R6, #0
    ADD R6, R6, -1
    STR R0, R6, #0
    
	JSR MULTIPLY

	; Pop return value and arguments
    LDR R0, R6, 0

	ADD R6, R6, 3

    ADD R3, R3, 1

    BR FOR

BASECASE1
    STR R0, R5, 3

TEARDOWN1
    ;; Tear down the stack
    LDR     R0, R6, #0      ;; Restore old R0
    LDR     R1, R6, #1      ;; Restore old R1
    LDR     R2, R6, #2      ;; Restore old R2
    LDR     R3, R6, #3      ;; Restore old R3
    LDR     R4, R6, #4      ;; Restore old R4
    ADD     R6, R5, #0      ;; Pop off restored registers and any local variables (LV)
    LDR     R5, R6, #1      ;; Restore old frame pointer (FP)
    LDR     R7, R6, #2      ;; Restore return address (RA)
    ADD     R6, R6, #3      ;; Pop off LV1, old FP, and RA
    RET

;; Needed to Simulate Subroutine Call in Complx
STACK .fill xF000
.end