import random 

SIZE = 9
NUMBERS_EASY = 50
NUMBERS_MEDIUM = 32
NUMBERS_HARD = 25

def isValid(board, row, col, num): # Ele não verifica o Sudoku por completo, ele so verifica se um número pode ser colocado ali
    # isValid(board, 4, 5, 7) Posso colocar o número 7 na linha 4, coluna 5?

    # Linha 
    for c in range(SIZE):
        if board[row][c] == num: 
            return False
    # Coluna
    for r in range(SIZE):
        if board[r][col] == num:
            return False

    # Bloco 3x3
    startRow = (row // 3) * 3 # Row = 7 --> 7//3 = 2 --> 3 * 2 = 6, A linha vai começar na posição 6
    startCol = (col // 3) * 3 # A Coluna tbm vai começar na posição 6

    for r in range(startRow, startRow + 3): # (6 , 9)
        for c in range(startCol, startCol + 3):
            if board[r][c] == num:
                return False
    return True

def findEmpty(board):
    for row in range(SIZE):
        for col in range(SIZE):
            if board[row][col] == 0:
                return row, col

    return None

def removerNumbers(board, amount):
    """
    positions = []
        for row in range(9):
            for col in range(9):
                positions.append((row, col))
    """
    positions = [ # Pega cada posição e add na Lista (0,0), (0,1)...
        (row, col)   # A cada rodada do for isso add o valor
        for row in range(9) 
        for col in range(9)
    ]

    # print(positions)

    random.shuffle(positions) # Embaralhar as posições para ter diferentes posicações retirada pra cada Sudoku
    currentNumbers = 81

    for row, col in positions: # Pega um posição aleatoria
        if currentNumbers <= amount: # Quando tiver menos que o ideal ele para
            break
        board[row][col] = 0
        currentNumbers -= 1
    return board

def generateSolution(board):
    empty = findEmpty(board)

    if empty is None:
        return True

    row, col = empty # Onde precisa colocar no Número
    numbers = list(range(1, 10)) 
    random.shuffle(numbers) # Garante que o Sudoku seja diferente um do outro

    for num in numbers:
        if isValid(board, row, col, num):  # Tentativa e ERRO se ela não achar a Sequencia certa ela volta pro começo
            board[row][col] = num
            if generateSolution(board):
                return True

            board[row][col] = 0

    return False

def generateSudoku(difficulty):
    """
    board = []
    for _ in range(9):
        row = []
        for _ in range(9):
            row.append(0)
        board.append(row)
    """
    board = [
        [0 for _ in range(9)] 
        for _ in range(9)
    ]
    # Gera solução completa
    generateSolution(board)

    # Quantidade de números
    if difficulty == 1:
        amount = NUMBERS_EASY
    elif difficulty == 2:
        amount = NUMBERS_MEDIUM
    elif difficulty == 3:
        amount = NUMBERS_HARD
    else:
        raise ValueError("Dificuldade inválida")

    # Remove números
    removerNumbers(board, amount)

    return board