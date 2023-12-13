;;=============================================================
;; CS 2110 - Spring 2023
;; Homework 5 - palindrome
;;=============================================================
;; Name: Vidit Pokharna
;;=============================================================

;;  NOTE: Let's decide to represent "true" as a 1 in memory and "false" as a 0 in memory.
;;
;;  Pseudocode (see PDF for explanation)
;;  Pseudocode values are based on the labels' default values.
;;
;;  String str = "aibohphobia";
;;  boolean isPalindrome = true
;;  int length = 0;
;;  while (str[length] != '\0') {
;;		length++;
;;	}
;; 	
;;	int left = 0
;;  int right = length - 1
;;  while(left < right) {
;;		if (str[left] != str[right]) {
;;			isPalindrome = false;
;;			break;
;;		}
;;		left++;
;;		right--;
;;	}
;;	mem[mem[ANSWERADDR]] = isPalindrome;

.orig x3000
	LD R0,STRING; R0 = STRING
	LD R1,ANSWERADDR; R1 = ANSWERADDR
	AND R2,R2,#0; R2 = isPalindrome = 0
	ADD R2,R2,#1; R2 = isPalindrome = 1
	AND R3,R3,#0; R3 = length = 0

WHILE1
	ADD R4,R0,R3; R4 = address of string[length]
    LDR R4,R4,#0; R4 = string[length]
	BRnz ENDWHILE1; while (str[length] != '\0')
	
	ADD R3,R3,#1; R3 = R3 + 1 = length + 1
	BR WHILE1; branch back to WHILE1

ENDWHILE1
	LD R0,STRING; R0 = STRING
	AND R4,R4,#0; R4 = left = 0
	AND R5,R5,#0; R5 = right = 0
	NOT R5,R5; R5 = right = -1
	ADD R5,R5,R3; R5 = right = length - 1

WHILE2
	NOT R6,R5; R6 = ~right
	ADD R6,R6,#1; R6 = -right
	ADD R6,R6,R4; R6 = R4 + R5 = left - right
	BRzp ENDWHILE2

IF
	ADD R6,R0,R4; R6 = address of string[left]
    LDR R6,R6,#0; R6 = string[left]
	ADD R7,R0,R5; R7 = address of string[right]
    LDR R7,R7,#0; R7 = string[right]
	NOT R7,R7; R7 = ~string[right]
	ADD R7,R7,#1; R7 = -string[right]
	ADD R7,R7,R6; R7 = string[left] - string[right]
	BRz ENDIF

	AND R2,R2,#0; R2 = isPalindrome = 0
	BR ENDWHILE2; break WHILE2

ENDIF
	ADD R4,R4,#1; R4 = left = left + 1
	AND R6,R6,#0; R6 = 0
	NOT R6,R6; R6 = -1
	ADD R5,R5,R6; R5 = right = right - 1
	BR WHILE2; branch to WHILE2

ENDWHILE2
	STR R2,R1,#0; mem[mem[ANSWERADDR]] = isPalindrome
	HALT

;; Do not change these values!
STRING	.fill x4004
ANSWERADDR 	.fill x5005
.end

;; Do not change any of the .orig lines!
.orig x4004				   
	.stringz "aibohphobia" ;; Feel free to change this string for debugging.
.end

.orig x5005
	ANSWER  .blkw 1
.end