import math

def showBoard(board, original, size):

    square = math.isqrt(size)

    # Largura de cada linha
    cell_width = 2

    # Borda superior
    print("┌" + "─" * (size * cell_width + square - 1) + "┐")
    for i, line in enumerate(board):

        print("│", end=" ")
        for j, num in enumerate(line):

            if num == 0:
                print("\033[90m.\033[0m", end=" ")
            elif original[i][j]:
                # Número original
                print(f"\033[92m{num}\033[0m", end=" ")
            else:
                # Número colocado pelo algoritmo
                print(f"\033[94m{num}\033[0m", end=" ")
            # Separação entre blocos
            if (j + 1) % square == 0 and j != size - 1:
                print("│", end=" ")

        print("│")
        # Separação entre blocos
        if (i + 1) % square == 0 and i != size - 1:
            print("├" + "─" * (size * cell_width + square - 1) + "┤")
    # Borda inferior
    print("└" + "─" * (size * cell_width + square - 1) + "┘")



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