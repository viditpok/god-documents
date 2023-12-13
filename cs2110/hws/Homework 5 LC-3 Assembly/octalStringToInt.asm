;;=============================================================
;; CS 2110 - Spring 2023
;; Homework 5 - octalStringToInt
;;=============================================================
;; Name: Vidit Pokharna
;;=============================================================

;;  Pseudocode (see PDF for explanation)
;;  Pseudocode values are based on the labels' default values.
;;
;;  String octalString = "2110";
;;  int length = 4;
;;  int value = 0;
;;  int i = 0;
;;  while (i < length) {
;;      int leftShifts = 3;
;;      while (leftShifts > 0) {
;;          value += value;
;;          leftShifts--;
;;      }
;;      int digit = octalString[i] - 48;
;;      value += digit;
;;      i++;
;;  }
;;  mem[mem[RESULTADDR]] = value;

.orig x3000
    LD R0,ASCII; R0 = ASCII = -48
    LD R1,OCTALSTRING; R1 = OCTALSTRING
    LD R2,LENGTH; R2 = LENGTH
    AND R3,R3,#0; R3 = VALUE = 0
    LD R4,RESULTADDR; R4 = RESULTADOR
    AND R5,R5,#0; R5 = i = 0

WHILE1
    NOT R6,R2; R6 = ~length
    ADD R6,R6,#1; R6 = -length
    ADD R6,R6,R5; R6 = i - length
    BRzp ENDWHILE1; while (i < length)

    AND R7,R7,#0; R7 = leftShifts = 0 
    ADD R7,R7,#3; R7 = leftShifts = 3

WHILE2
    ADD R7,R7,#0; setting condition code
    BRnz ENDWHILE2

    ADD R3,R3,R3; R3 = value + value
    AND R6,R6,#0; R6 = 0
    NOT R6,R6; R6 = -1
    ADD R7,R7,R6; R7 = R7 - 1 = leftshifts - 1
    BR WHILE2

ENDWHILE2
    ADD R1,R1,R5; R1 = digit = address of octalstring[i]
    LDR R1,R1,#0; R1 = digit = octalstring[i]
    ADD R1,R1,R0; R1 = digit = octalstring[i] - 48
    ADD R3,R3,R1; R3 = value + digit
    ADD R5,R5,#1; i = i + 1
    LD R1,OCTALSTRING; R1 = OCTALSTRING
    BR WHILE1; branch back to WHILE1

ENDWHILE1
    STR R3,R4,#0; mem[mem[RESULTADDR]] = value;
    HALT
    
;; Do not change these values! 
;; Notice we wrote some values in hex this time. Maybe those values should be treated as addresses?
ASCII           .fill -48
OCTALSTRING     .fill x5000
LENGTH          .fill 4
RESULTADDR      .fill x4000
.end

.orig x5000                    ;;  Don't change the .orig statement
    .stringz "2110"            ;;  You can change this string for debugging!
.end
