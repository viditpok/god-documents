;;=============================================================
;; CS 2110 - Spring 2023
;; Homework 5 - buildMaxArray
;;=============================================================
;; Name: Vidit Pokharna
;;=============================================================

;;  Pseudocode (see PDF for explanation)
;;  Pseudocode values are based on the labels' default values.
;;
;;	int A[] = {-2, 2, 1};
;;	int B[] = {1, 0, 3};
;;	int C[3];
;;	int length = 3;
;;
;;	int i = 0;
;;	while (i < length) {
;;		if (A[i] >= B[length - i - 1]) {
;;			C[i] = 1;
;;		}
;;		else {
;;			C[i] = 0;
;;		}
;;		i++;
;;	}

.orig x3000
    LD R0,A; R0 = A[]
    LD R1,B; R1 = B[]
    LD R2,C; R2 = C[]
    LD R3,LENGTH; R3 = length
    AND R4,R4,#0; R4 = i = 0

WHILE
    NOT R5,R3; R5 = ~length
    ADD R5,R5,#1; R5 = -length
    ADD R5,R5,R4; R5 = i - length
    BRzp ENDWHILE; while (i - length < 0)

IF
    ADD R0,R0,R4; R0 = address of A[i]
    LDR R0,R0,#0; R0 = A[i]
    NOT R6,R5; R6 = length - i - 1
    ADD R7,R1,R6; R7 = address of B[length - i - 1]
    LDR R7,R7,#0; R7 = B[length - i - 1]
    NOT R7,R7; R7 = ~B[length - i - 1]
    ADD R7,R7,#1; R7 = -B[length - i - 1]
    ADD R7,R7,R0; R7 = A[i] - B[length - i - 1]
    BRn ELSE

    ADD R2,R2,R4; R0 = C[i]
    AND R6,R6,#0; R6 = 0
    ADD R6,R6,#1; R6 = 1
    STR R6,R2,#0; C[i] = 1
    BR ENDIF
ELSE
    ADD R2,R2,R4; R0 = C[i]
    AND R6,R6,#0; R6 = 0
    STR R6,R2,#0; C[i] = 0

ENDIF
    ADD R4,R4,#1; R4 = i + 1
    LD R0,A; R0 = A[]
    LD R1,B; R1 = B[]
    LD R2,C; R2 = C[]
    BR WHILE
    
ENDWHILE
    HALT

;; Do not change these addresses! 
;; We populate A and B and reserve space for C at these specific addresses in the orig statements below.
A 		.fill x3200		
B 		.fill x3300		
C 		.fill x3400		
LENGTH 	.fill 3			;; Change this value if you decide to increase the size of the arrays below.
.end

;; Do not change any of the .orig lines!
;; If you decide to add more values for debugging, make sure to adjust LENGTH and .blkw 3 accordingly.
.orig x3200				;; Array A : Feel free to change or add values for debugging.
	.fill -2
	.fill 2
	.fill 1
.end

.orig x3300				;; Array B : Feel free change or add values for debugging.
	.fill 1
	.fill 0
	.fill 3
.end

.orig x3400
	.blkw 3				;; Array C: Make sure to increase block size if you've added more values to Arrays A and B!
.end