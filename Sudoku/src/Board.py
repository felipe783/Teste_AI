import math
import random

def _blockDims(size):
    """Encontra (block_rows, block_cols) tal que block_rows * block_cols == size,
    escolhendo a divisão mais próxima de um quadrado (funciona para 9, 16, 6, 12...)."""
    best = (1, size)
    for i in range(1, math.isqrt(size) + 1):
        if size % i == 0:
            best = (i, size // i)
    return best  # ex: 9 -> (3,3) | 6 -> (2,3) | 12 -> (3,4) | 16 -> (4,4)

def showBoard(board, original, size):
    block_rows, block_cols = _blockDims(size)
    digit_width = len(str(size))   # largura necessária pro maior número possível
    cell_width = digit_width + 1   # número + 1 espaço

    def buildRowPlain():
        parts = []
        for col in range(size):
            parts.append(" " * cell_width)
            if (col + 1) % block_cols == 0 and col != size - 1:
                parts.append("│ ")
        return "".join(parts)

    content_width = len(buildRowPlain())
    top = "┌" + "─" * (content_width + 1) + "┐"
    mid = "├" + "─" * (content_width + 1) + "┤"
    bot = "└" + "─" * (content_width + 1) + "┘"

    print(top)
    for i, line in enumerate(board):
        row = "│ "
        for j, num in enumerate(line):
            text = str(num).rjust(digit_width) if num != 0 else ".".rjust(digit_width)
            if num == 0:
                cell = f"\033[90m{text}\033[0m "
            elif original[i][j]:
                cell = f"\033[92m{text}\033[0m "
            else:
                cell = f"\033[94m{text}\033[0m "
            row += cell
            if (j + 1) % block_cols == 0 and j != size - 1:
                row += "│ "
        row += "│"
        print(row)

        if (i + 1) % block_rows == 0 and i != size - 1:
            print(mid)
    print(bot)


def checkVictory(board, size):  

    square = math.isqrt(size)

    for row in board:  # Linha
        # Pega uma Lista(row) e comprime em outra lista(numbers)
        numbers = [num for num in row]  # Pega os Valores por linha, "numbers[1,2...9]""
        if sorted(numbers) != list(range(1, size + 1)): # Organiza os Numeros e ve se possuia os numeros de 1 a 9
            return False

    # Precisa buscar cada celula em cada linha 
    for column in range(size):  # Coluna
        numbers = [board[row][column] for row in range(size)]
        if sorted(numbers) != list(range(1, size + 1)):
            return False

    # 3x3
    # TODO: O Sudoku é como um jogo da velha com "mini" jogos da velha dentro dele
    for row_start in range(0, size, square): # Começa no 0, termina no 9, incremento de 3(0,3,6)
        for column_start in range(0, size, square): # Coluna
            numbers = []
            
            for row in range(row_start, row_start + square):
                for column in range(column_start, column_start + square):
                    numbers.append(board[row][column])
        
            if sorted(numbers) != list(range(1, size + 1)):
                return False
    return True