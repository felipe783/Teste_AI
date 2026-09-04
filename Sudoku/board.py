def showBoard(board):
    print("┌───────┬───────┬───────┐")
    for i, line in enumerate(board[0]):
        print("│", end=" ")
        for j, cell in enumerate(line):
            if cell == 0:
                print("\033[90m.\033[0m", end=" ")
            else:
                print(f"\033[92m{cell}\033[0m", end=" ")
            if j == 2 or j == 5:
                print("│", end=" ")
        print("│")
        if i == 2 or i == 5:
            print("├───────┼───────┼───────┤")
    print("└───────┴───────┴───────┘")

def checkVictory(board):  

    for row in board[0]:  # Linha
        # Pega uma Lista(row) e comprime em outra lista(numbers)
        numbers = [Cell for Cell in row]  # Pega os Valores por linha, "numbers[1,2...9]""
        if sorted(numbers) != list(range(1, 10)): # Organiza os Numeros e ve se possuia os numeros de 1 a 9
            return False

    # Precisa buscar cada celula em cada linha 
    for column in range(9):  # Coluna
        numbers = [board[0][row][column] for row in range(9)]
        if sorted(numbers) != list(range(1, 10)):
            return False

    # 3x3
    # TODO: O Sudoku é como um jogo da velha com "mini" jogos da velha dentro dele
    for row_start in range(0, 9, 3): # Começa no 0, termina no 9, incremento de 3(0,3,6)
        for column_start in range(0, 9, 3): # Coluna
            numbers = []
            
            for row in range(row_start, row_start + 3):
                for column in range(column_start, column_start + 3):
                    numbers.append(board[0][row][column])
        
            if sorted(numbers) != list(range(1, 10)):
                return False
    return True
    """
    for row_start in range(0, 9, 3):
        for column_start in range(0, 9, 3):

            numbers = [
                board[row][column].value
                for row in range(row_start, row_start + 3)
                for column in range(column_start, column_start + 3)
            ]

            if sorted(numbers) != list(range(1, 10)):
                return False
    return True
    """
