import random
import core.config as core
import copy


GRID_SIZE = 33

def createGrid():

    grid = [
        [None for _ in range(GRID_SIZE)]
        for _ in range(GRID_SIZE)
    ]

    for _, (row, col) in core.SAMURAI_CONFIG:

        for i in range(9):
            for j in range(9):

                grid[row + i][col + j] = 0

    return grid


def findEmpty(board):
    # Heuristica MRV(Minimum Remaining Values)

    minOptions = 10
    bestCell = None
    bestValidNumbers = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            if board[row][col] == 0:
                validNumbers = [num for num in range(1, 10) if isValidSamurai(board, row, col, num)]
                numOptions = len(validNumbers)

                if numOptions == 0:
                    return row, col, []
                if numOptions < minOptions:
                    minOptions = numOptions
                    bestCell = (row, col)
                    bestValidNumbers = validNumbers

                    if minOptions == 1:
                        return bestCell[0], bestCell[1], bestValidNumbers
    if bestCell:
        return bestCell[0], bestCell[1], bestValidNumbers
    return None, None, None


def isValid(board, row, col, num, startRow, startCol):

    # Linha
    for c in range(startCol, startCol + 9):
 
        if board[row][c] == num:
            return False

    # Coluna
    for r in range(startRow, startRow + 9):

        if board[r][col] == num:
            return False

    # Bloco 3x3
    localRow = row - startRow
    localCol = col - startCol

    blockRow = startRow + (localRow // 3) * 3
    blockCol = startCol + (localCol // 3) * 3

    for r in range(blockRow, blockRow + 3):
        for c in range(blockCol, blockCol + 3):

            if board[r][c] == num:
                return False

    return True


def isValidSamurai(board, row, col, num):
    for _, (startRow, startCol) in core.SAMURAI_CONFIG:

        # Verifica se a posição pertence a este Sudoku
        if (startRow <= row < startRow + 9 and startCol <= col < startCol + 9):
            if not isValid(board,row,col,num,startRow,startCol):
                return False

    return True

def isValid(board, row, col, num, startRow, startCol):

    # Linha
    for c in range(startCol, startCol + 9):

        if board[row][c] == num:
            return False

    # Coluna
    for r in range(startRow, startRow + 9):

        if board[r][col] == num:
            return False

    # Bloco 3x3
    localRow = row - startRow
    localCol = col - startCol

    blockRow = startRow + (localRow // 3) * 3
    blockCol = startCol + (localCol // 3) * 3

    for r in range(blockRow, blockRow + 3):
        for c in range(blockCol, blockCol + 3):

            if board[r][c] == num:
                return False

    return True

def generateSolution(board):
    row, col, validNums = findEmpty(board)

    # Se não tem mais células, resolveu
    if row is None:
        return True
    
    # Se encontrou uma célula sem nenhuma opção válida 
    if not validNums:
        return False

    # Embaralha apenas os números que já sabemos que são válidos para aquela posição
    random.shuffle(validNums)

    for num in validNums:
        board[row][col] = num

        if generateSolution(board):
            return True

        board[row][col] = 0

    return False

def removerNumbers(board, amount):
    positions = []

    # Mapeia apenas as posições do tabuleiro que possuem números gerados
    for row in range(len(board)): # Varrerá de 0 a 32 (tamanho 33)
        for col in range(len(board[0])):
            # Verifica se a célula faz parte do jogo e já está preenchida
            if board[row][col] != 0 and board[row][col] is not None:
                positions.append((row, col))

    random.shuffle(positions) 
    
    # Descobre quantos números totais o tabuleiro tem agora (em vez de fixar em 81)
    currentNumbers = len(positions)

    # Remove os números um por um
    for row, col in positions:
        # Quando a quantidade de números restantes chegar ao 'amount' (nível de dificuldade), ele para
        if currentNumbers <= amount: 
            break
            
        board[row][col] = 0
        currentNumbers -= 1

    return board

def generateSamurai():

    while True:
        grid = createGrid()

        print("Gerando Samurai...")
        
        amount = core.SAMURAI_NUMBERS_HARD

        if generateSolution(grid):
            solveGrid = copy.deepcopy(grid)
            puzzleGrid = removerNumbers(grid, amount)

            return solveGrid, puzzleGrid


solveGrid, puzzleGrid = generateSamurai()
# print(solveGrid)

for row in puzzleGrid:
    print(
        " ".join(
            "." if value is None else " " if value == 0 else str(value)
            for value in row
        )
    )