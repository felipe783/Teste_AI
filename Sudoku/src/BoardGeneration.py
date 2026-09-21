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

    for r in range(startRow, startRow + square): # ( 6 , 9)
        for c in range(startCol, startCol + square): # ( 6, 9 )
            if board[r][c] == num:
                return False
    return True # é valido o Número ali

def findEmpty(board, size):
    minOptions = size + 1
    bestCell = None
    bestCandidates = []

    for row in range(size):
        for col in range(size):
            if board[row][col] == 0: # So as vazias

                candidates = [
                    # Se num no range de size + 1 pode ser colocado na Celula
                    # List comprehension, [item for item in iterável if ...] 
                    num # ADD o número em cadidates
                    for num in range(1, size + 1)
                    if isValid(board, row, col, num, size)
                ]

                # Nenhuma possibilidade a Celula
                if not candidates:
                    return row, col, [] # Sudoku impossivel
                if len(candidates) < minOptions: # Sempre guarda a Menor celula
                    minOptions = len(candidates) 
                    bestCell = (row, col) # é a melhor celula bestcell = (1 , 2)
                    bestCandidates = candidates 

                    if minOptions == 1: # Nao existe Celula melhor que uma com 1 candidato
                        return row, col, candidates # (1 linhas, 2 colunas , [3 , 4])

    # Nao existem Celula vazias
    if bestCell is None:
        return None

    return bestCell[0], bestCell[1], bestCandidates

def removerNumbers(board, amount, size):
    positions = [
        (row, col)
        for row in range(size)
        for col in range(size)
        if board[row][col] != 0 and board[row][col] is not None # Todas que estão preenchidas
    ]

    random.shuffle(positions) # Randomizar o número pego
    currentNumbers = len(positions)

    for row, col in positions: # positon = [(1,2), (7,8) , (3,9)..... (linha N, coluna N)]
        if currentNumbers <= amount: # Quando tiver menos que o ideal ele para
            break
        board[row][col] = 0
        currentNumbers -= 1
    return board

def generateSudoku(removeCells, size):

    board = [ # Criar o Board so com 0
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

def generateSolution(board,size):
    # print("Gerando")
    empty = findEmpty(board, size) 
    
    if empty is None: # Jogo finalizado
        return True

    row, col, candidates = empty # (linha X, coluna X , [ Candidato 1 ... Canditado N ])

    if not candidates: # Impossivel
        return False

    random.shuffle(candidates) # Pegar um Canditado aleatorio

    for num in candidates:
        board[row][col] = num

        if generateSolution(board, size):
            return True # Finalizar se achou a solução
        
        board[row][col] = 0

    return False

"""def _blockDims(size):
    best = (1, size)
    for i in range(1, math.isqrt(size) + 1):
        if size % i == 0:
            best = (i, size // i)
    return best  # ex: 9 -> (3,3) | 81 -> (9,9) | 6 -> (2,3)


def generateSolution(board, size):
    R, C = _blockDims(size)

    def pattern(r, c):
        return (C * (r % R) + r // R + c) % size

    def shuffled(seq):
        seq = list(seq)
        random.shuffle(seq)
        return seq

    rows = [g * R + r for g in shuffled(range(C)) for r in shuffled(range(R))]
    cols = [g * C + c for g in shuffled(range(R)) for c in shuffled(range(C))]
    nums = shuffled(range(1, size + 1))

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            board[i][j] = nums[pattern(r, c)]

    return True

def isValidSolution(board, size):
    R, C = _blockDims(size)
    expected = set(range(1, size + 1))

    for row in board:
        if set(row) != expected:
            return False

    for c in range(size):
        col = [board[r][c] for r in range(size)]
        if set(col) != expected:
            return False

    for br in range(0, size, R):
        for bc in range(0, size, C):
            block = [
                board[r][c]
                for r in range(br, br + R)
                for c in range(bc, bc + C)
            ]
            if set(block) != expected:
                return False

    return True
"""