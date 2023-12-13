!============================================================
! CS 2200 Homework 2 Part 2: Tower of Hanoi
!
! Apart from initializing the stack,
! please do not edit mains functionality. You do not need to
! save the return address before jumping to hanoi in
! main.
!============================================================

main:
    lea     $sp, stack              ! load the label address into $sp
    lw      $sp, 0($sp)             ! load $sp into $sp

    lea     $at, hanoi              ! loads address of hanoi label into $at

    lea     $a0, testNumDisks2      ! loads address of testNumDisks2 into $a0
    lw      $a0, 0($a0)             ! loads value of $a0 into $a0

    jalr    $at, $ra                ! hanoi()
    halt                            ! halt when we return

hanoi:
    addi    $sp, $sp, -4            ! allocates space on stack
    sw      $fp, 0($sp)             ! store $fp into memory location at current $sp
    addi    $fp, $sp, 0             ! store $sp into $fp

    addi    $sp, $sp, -4            ! allocates space on stack
    sw      $t0, 0($sp)             ! store $t0 in $sp
    addi    $t0, $zero, 1           ! $t0 = 1
    beq     $a0, $t0, base          ! if $a0 == 1, go to base
    beq     $zero, $zero, else      ! go to else

else:
    addi    $a0, $a0, -1            ! n--
    addi    $sp, $sp, -4            ! allocates space on stack
    sw      $ra, 0($sp)             ! store $ra in $sp
    jalr    $at, $ra                ! hanoi()
    lw      $ra, 0($sp)             ! load $sp into $ra
    addi    $sp, $sp, 4             ! pop 4 spots off stack

    add     $v0, $v0, $v0           ! $v0 *= 2
    addi    $v0, $v0, 1             ! $v0++
    beq     $zero, $zero, teardown  ! go to teardown

base:
    addi     $v0, $zero, 1          ! return 1
    beq      $zero, $zero, teardown ! go to teardown

teardown:
    lw      $fp, 0($sp)             ! restore old frame pointer
    addi    $sp, $sp, 4             ! pop 4 spots off stack
    lw      $t0, 0($sp)             ! load $sp into $t0
    addi    $sp, $sp, 4             ! pop 4 spots off stack
    jalr    $ra, $zero              ! return to caller

stack: .word 0xFFFF                 ! the stack begins here

! Words for testing \/

! 1
testNumDisks1:
    .word 0x0001

! 10
testNumDisks2:
    .word 0x000a

! 20
testNumDisks3:
    .word 0x0014
