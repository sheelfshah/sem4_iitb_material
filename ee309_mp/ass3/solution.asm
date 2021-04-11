;setting addresses for vars
s2 EQU [100H]
s1 EQU [102H]
n  EQU [104h]

MOV SP, 0DF20H; as given in question

;near is used for small model
JMP NEAR PTR main

subroutine:

PUSH BP
MOV BP, SP            ;move SP to BP

;preserving registers            
PUSH CX
PUSH SI
PUSH DI

;retrieving n,s1,s2 from the correct relative locations, and storing them in CX, SI, DI
; a 2 is added to account for the BP pus
MOV CX, [BP+4]
MOV SI, [BP+6]
MOV DI, [BP+8]

; copying the strings till CX is decremented to 0
REP MOVSB

;retrieving registers
POP DI
POP SI
POP CX
MOV SP, BP
POP BP

                       
RET 6;return from function and add 6 to SP so that all the 3 arguments are removed

main:

;pushing variables
PUSH s2              
PUSH s1
PUSH n

CALL subroutine      ;call our function

HLT