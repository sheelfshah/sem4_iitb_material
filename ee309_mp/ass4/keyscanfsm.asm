# keyboard scanning function using
# FSM discussed in class

# some instructions are implemented by the assembler (eg. la)
# since they seemed standard, I have used them


# s0 stores state
# s1 stores result of test (1/0)
# s2 stores value of pressed key
# s3 stores key index for buffer

# fsm state diagram, based on state number
             .data
Test_Tab:    .word 0, 1, 1, 1
Yes_Actions: .word 1, 2, 0, 0
No_Actions:  .word 0, 0, 0, 0
Yes_Next:    .word 1, 2, 2, 2
No_Next:     .word 0, 0, 3, 0
P0:          .word 0x40000000

# state variables
Cur_St:      .byte 0x00
Test_Result: .byte 0x00
Key_Code:    .byte 0x7E
Key_Index:   .byte 0 

Test_Jmp:    .word AnyKey, TheKey
Action_Jmp:  .word DoNothing, FindKey, ReportKey

KeyBuffer:   .space 16

# assume all t variables are local and overwriteable
.text
.globl main

main:


KeyScan:
    # callable function for keyboard scanning
    # will affect s0, ... s3, t1, t2

    # since all 4 s registers will be required
    # it is preferable to directly change them instead of
    # taking them as arguments and then returning them

    addi $sp,$sp,-4     # Adjust stack pointer
    sw $s0,0($sp)       # Save $s0
    addi $sp,$sp,-4     # Adjust stack pointer
    sw $s1,0($sp)       # Save $s1
    addi $sp,$sp,-4     # Adjust stack pointer
    sw $s2,0($sp)       # Save $s2
    addi $sp,$sp,-4     # Adjust stack pointer
    sw $s3,0($sp)       # Save $s3

    addi $sp,$sp,-4     # Adjust stack pointer
    sw $ra,0($sp)       # Save $ra

    jal Init
    nop
    jal DoTest
    nop
    jal DoAction
    nop
    jal UpdateState
    nop

    sb $s0, Cur_St      # store in Cur_St
    sb $s1, Test_Result # store in Test_Result
    sb $s2, Key_Code    # store in Key_Code
    sb $s3, Key_Index   # store in Key_Index

    lw $ra,0($sp)       # Restore $ra
    addi $sp,$sp,4      # Adjust stack pointer

    lw $s3,0($sp)       # Restore $s3
    addi $sp,$sp,4      # Adjust stack pointer
    lw $s2,0($sp)       # Restore $s2
    addi $sp,$sp,4      # Adjust stack pointer
    lw $s1,0($sp)       # Restore $s1
    addi $sp,$sp,4      # Adjust stack pointer
    lw $s0,0($sp)       # Restore $s0
    addi $sp,$sp,4      # Adjust stack pointer

    jr $ra              # return to main program
    nop


Init:
    # initializing state and key index to 0
    add $s0, $zero, $zero
    add $s3, $zero, $zero
    jr $ra                 # return to KeyScan
    nop
    

DoTest:
    # called by KeyScan
    # jumps to a test, and the test returns back to KeyScan
    # assumes s0 has current state

    la $t1, Test_Tab
    sll $t2,$s0,2       # $t2=index*4
    addu $t1, $t1, $t2  # increment t1 aptly
    lw $t1, ($t1)       # value at t1 stored in t1
    la $t0, Test_Jmp    # $t0=base address of the jump table
    add $t1,$t1,$t0     # $t1+$t0 = actual address of jump label
    lw  $t1,($t1)       # <-- load target address 
    jr  $t1             # jump to test
    nop


TheKey:
    addi $t1, $zero, 0x0F       # t1 = 0FH
    sb $t1, P0                 # move 0F to P0
    lbu $t1, P0                 # read P0
    # if a key is pressed, one column gets pulled to zero
    # now make all rows one, keeping columns as they are
    ori $t1, $t1, 0xF0
    sb $t1, P0                 # move FX to P0
    lbu $t1, P0                 # read P0

    bne $t1, $s2, the_key_not_pressed
    addi $s1, $zero, 1          # set s1
    jr $ra                      # return to KeyScan
    nop
    the_key_not_pressed: 
        add $s1, $zero, $zero   # clear s1
        jr $ra                  # return to KeyScan
        nop


AnyKey:
    addi $t1, $zero, 0x0F   # t1 = 0FH
    sb $t1, P0             # move 0F to P0
    lbu $t1, P0             # read P0
    # if a key is pressed, one column gets pulled to zero, and therefore t1<0F
    slti $s1, $t1, 0x0F     # s1 = 1 if any key pressed, else 0
    jr $ra                  # return to KeyScan
    nop


DoAction:
    # called by KeyScan
    # jumps to an action, and the action returns back to KeyScan
    # assumes s0 has current state and s1 has result of test

    beq $s1, $zero, do_no_action
    # if test results no, do no_actions, else do yes_actions

    la $t1, Yes_Actions
    sll $t2,$s0,2           # $t2=index*4
    addu $t1, $t1, $t2      # increment t1 aptly
    lw $t1, ($t1)           # value at t1 stored in t1
    la $t0, Action_Jmp      # $t0=base address of the jump table
    add $t1,$t1,$t0         # $t1+$t0 = actual address of jump label
    lw  $t1,($t1)           # <-- load target address 
    jr  $t1                 # jump to action
    nop

    do_no_action:
        la $t1, No_Actions
        sll $t2,$s0,2       # $t2=index*4
        addu $t1, $t1, $t2  # increment t1 aptly
        lw $t1, ($t1)       # value at t1 stored in t1
        la $t0, Action_Jmp  # $t0=base address of the jump table
        add $t1,$t1,$t0     # $t1+$t0 = actual address of jump label
        lw  $t1,($t1)       # <-- load target address 
        jr  $t1             # jump to action
        nop


DoNothing:
    jr $ra                  # return to KeyScan
    nop


FindKey:
    addi $t1, $zero, 0x0F       # t1 = 0FH
    sb $t1, P0                 # move 0F to P0
    lbu $t1, P0                 # read P0
    # if a key is pressed, one column gets pulled to zero
    # now make all rows one, keeping columns as they are
    ori $t1, $t1, 0xF0
    sb $t1, P0                 # move FX to P0
    lbu $t1, P0                 # read P0
    add $s2, $t1, $zero         # s2 is set as key_pressed
    jr $ra                      # return to KeyScan
    nop


ReportKey:
    la $t1, KeyBuffer
    sll $t2,$s3,2       # $t2=index*4
    addu $t1, $t1, $t2  # increment t1 aptly
    sb $s2, ($t1)      # pressed_key stored in (t1)
    addi $s3, $s3, 1    # increment key index
    andi $s3, $s3, 0x07 # reset to 0 if index=8 
    jr  $ra             # return to KeyScan
    nop


UpdateState:
    # called by KeyScan
    # updates state(s0) and then returns back to KeyScan
    # assumes s0 has current state and s1 has result of test

    beq $s1, $zero, use_no_states
    # if test results no, use no_states, else use yes_states

    la $t1, Yes_Next
    sll $t2,$s0,2           # $t2=index*4
    addu $t1, $t1, $t2      # increment t1 aptly
    lw $s0, ($t1)           # value at t1 stored in s0 
    jr  $ra                 # return to KeyScan
    nop

    use_no_states:
        la $t1, No_Next
        sll $t2,$s0,2       # $t2=index*4
        addu $t1, $t1, $t2  # increment t1 aptly
        lw $s0, ($t1)       # value at t1 stored in s0 
        jr  $ra             # return to KeyScan
        nop