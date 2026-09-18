import random
import core.config as core


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


def findEmpty(board, startRow, startCol):

    for row in range(startRow, startRow + 9):
        for col in range(startCol, startCol + 9):

            if board[row][col] == 0:
                return row, col

    return None


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


def generateSolution(board, startRow, startCol):

    empty = findEmpty(board, startRow, startCol)

    if empty is None:
        return True

    row, col = empty

    numbers = list(range(1, 10))
    random.shuffle(numbers)

    for num in numbers:

        if isValid(
            board,
            row,
            col,
            num,
            startRow,
            startCol
        ):

            board[row][col] = num

            if generateSolution(board, startRow, startCol):
                return True

            board[row][col] = 0

    return False


def generateSamurai():

    while True:

        grid = createGrid()

        sucesso = True

        for nome, (row, col) in core.SAMURAI_CONFIG:

            print(f"Gerando {nome}...")

            if not generateSolution(grid, row, col):

                sucesso = False
                break

        if sucesso:
            return grid


grid = generateSamurai()

for row in grid:

    print(
        " ".join(
            "." if value is None else str(value)
            for value in row
        )
    )