import random 
import math

def isValid(board, row, col, num, size): # Ele não verifica o Sudoku por completo, ele so verifica se um número pode ser colocado ali
    # isValid(board, 4, 5, 7) Posso colocar o número 7 na linha 4, coluna 5?

    square = math.isqrt(size)
    # Linha 
    for c in range(size):
        if board[row][c] == num: 
            return False
    # Coluna
    for r in range(size):
        if board[r][col] == num:
            return False

    # Bloco 3x3
    # Ex: square = 3
    startRow = (row // square) * square # Row = 7 --> 7//3 = 2 --> 3 * 2 = 6, A linha vai começar na posição 6
    startCol = (col // square) * square # A Coluna tbm vai começar na posição 6

    for r in range(startRow, startRow + square): # (6 , 9)
        for c in range(startCol, startCol + square):
            if board[r][c] == num:
                return False
    return True

def findEmpty(board, size):
    minOptions = size + 1
    bestCell = None
    bestCandidates = []

    for row in range(size):
        for col in range(size):
            if board[row][col] == 0:

                candidates = [
                    num
                    for num in range(1, size + 1)
                    if isValid(board, row, col, num, size)
                ]

                # Nenhuma possibilidade para essa célula
                if not candidates:
                    return row, col, []
                if len(candidates) < minOptions:
                    minOptions = len(candidates)
                    bestCell = (row, col)
                    bestCandidates = candidates
                    # Não existe célula melhor que uma com 1 candidato
                    if minOptions == 1:
                        return row, col, candidates

    # Não existem células vazias
    if bestCell is None:
        return None

    return bestCell[0], bestCell[1], bestCandidates

def removerNumbers(board, amount, size):
    positions = [
        (row, col)
        for row in range(size)
        for col in range(size)
        if board[row][col] != 0 and board[row][col] is not None
    ]

    random.shuffle(positions)

    currentNumbers = len(positions)

    for row, col in positions: # Pega um posição aleatoria
        if currentNumbers <= amount: # Quando tiver menos que o ideal ele para
            break
        board[row][col] = 0
        currentNumbers -= 1
    return board

def generateSolution(board,size):
    # print("Gerando")
    empty = findEmpty(board, size)
    
    if empty is None:
        return True

    row, col, candidates = empty 
    if not candidates:
        return False

    random.shuffle(candidates)

    for num in candidates:
        board[row][col] = num

        if generateSolution(board, size):
            return True
        board[row][col] = 0

    return False

def generateSudoku(removeCells, size):

    board = [
        [0 for _ in range(size)] 
        for _ in range(size)
    ]
    # Gera solução completa
    generateSolution(board, size)

    # Quantidade de números
    amount = removeCells
    # print(f"size={size}, amount={amount}")
    # Remove números
    removerNumbers(board, amount, size)

    return board