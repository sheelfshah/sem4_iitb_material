ORG 100H

;ASSIGNING MEMORY LOCATIONS
DATA_1 EQU [50]
DATA_2 EQU [51]
RESULT_1 EQU [52] 
RESULT_2 EQU [53]
RESULT_3 EQU [54]
RESULT_4 EQU [55] 

MOV DATA_1, 52H; ROLL NO. IS 19D070052
MOV DATA_2, 79H; GIVEN IN QUESTION

;SUBTRACTION
MOV AL, DATA_1
SUB AL, DATA_2; PERFORM SUBTRACTION
DAS; DECIMAL ADJUST
MOV DL, AL
;RETRIEVE NIBBLES AND STORE IN APT LOCATION
AND DL, 00001111B
AND AL, 11110000B
SHR AL, 4
MOV RESULT_1, AL
MOV RESULT_2, DL

;MULTIPLICATION
MUL DL; PERFORM MULTIPLICATION
MOV DX, AX
AAM; DECIMAL ADJUST
;NO NEED TO SEPARATE NIBBLES
MOV RESULT_3, AH
MOV RESULT_4, AL

;DIVISION
MOV AX, DX
MOV DL, 7
DIV DL; PERFORM DIVISION
;NO NEED TO SEPARATE NIBBLES
MOV RESULT_3, AH
MOV RESULT_4, AL

HLT