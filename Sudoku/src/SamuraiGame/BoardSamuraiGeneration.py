import random
import core.config as core

class Cor:
    RESET = "\033[0m"
    CINZA = "\033[90m"

def createGridSamurai():
    grid = [[None for _ in range(core.GRID_SIZE)] for _ in range(core.GRID_SIZE)]
    for _, (row, col) in core.SAMURAI_CONFIG:
        for i in range(9):
            for j in range(9):
                grid[row + i][col + j] = 0
    return grid

def isValidSamuraiSub(board, row, col, num, startRow, startCol):
    # Linha
    for c in range(startCol, startCol + 9):
        if board[row][c] == num:
            return False
    # Coluna
    for r in range(startRow, startRow + 9):
        if board[r][col] == num:
            return False
    # Bloco 3x3
    localRow, localCol = row - startRow, col - startCol
    blockRow, blockCol = startRow + (localRow // 3) * 3, startCol + (localCol // 3) * 3
    for r in range(blockRow, blockRow + 3):
        for c in range(blockCol, blockCol + 3):
            if board[r][c] == num:
                return False
    return True

def isValidSamuraiGlobally(board, row, col, num):
    for _, (startRow, startCol) in core.SAMURAI_CONFIG:
        if (startRow <= row < startRow + 9 and startCol <= col < startCol + 9):
            if not isValidSamuraiSub(board, row, col, num, startRow, startCol):
                return False
    return True

def findEmptySamurai(board):
    minOptions = 10
    bestCell = None
    bestValidNumbers = []

    for row in range(core.GRID_SIZE):
        for col in range(core.GRID_SIZE):
            if board[row][col] == 0:
                validNumbers = [num for num in range(1, 10) if isValidSamuraiGlobally(board, row, col, num)]
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

def solveSamurai_Backtracking(board):
    row, col, validNums = findEmptySamurai(board)
    if row is None:
        return True
    if not validNums:
        return False
        
    random.shuffle(validNums)
    for num in validNums:
        board[row][col] = num
        if solveSamurai_Backtracking(board):
            return True
        board[row][col] = 0
    return False

def removeNumbersSamurai(board, amount):
    positions = []

    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] != 0 and board[row][col] is not None:
                positions.append((row, col))
                
    random.shuffle(positions) 
    currentNumbers = len(positions)

    for row, col in positions:
        if currentNumbers <= amount: 
            break
        board[row][col] = 0
        currentNumbers -= 1
    return board

def generateSamurai(removeCells):
    print(f"\n{Cor.CINZA}Gerando tabuleiro Samurai (pode levar alguns segundos)...{Cor.RESET}")
    grid = createGridSamurai()
    amount = removeCells
    # Mapeando dificuldade para a quantidade de números que sobram (exemplo)
    if solveSamurai_Backtracking(grid):
        puzzleGrid = removeNumbersSamurai(grid, amount)
        return puzzleGrid
    return grid