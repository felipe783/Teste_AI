from src.Board import *
from src.BoardGeneration import *
from src.Probability import *
from src.Logs import *

import time

numBoards = int(input("Fale o número de Jogos que deseja: "))
print("\n---------------\n")
print("1 - Fácil")
print("2 - Médio")
print("3 - Dificil")
print("\n---------------\n")
difficulty = int(input("Dificuldade: "))

inicio = time.time()

solved = 0
impossible = 0
totalQuadrants = 0
totalCoverage = 0

for i in range(numBoards):
    print(f"Jogo {i + 1}")
    board = generateSudoku(difficulty)
    original = [
        [num != 0 for num in row]
        for row in board
    ]

    print("Board Inicial:")
    showBoard(board, original)

    result = solveProbabilistic(board)

    print("Board Final")
    showBoard(board, original)

    # Salvar no JSON
    correctQuadrants = checkQuadrants(board)
    coverage = (correctQuadrants / 9) * 100

    totalQuadrants += correctQuadrants
    totalCoverage += coverage

    if result:
        solved += 1
    else:
        impossible += 1

    print(f"Quadrantes corretos: {correctQuadrants}/9")
    print(f"Cobertura: {coverage:.2f}%")
    print()

fim = time.time()
execution_time = fim - inicio

saveResults(
    difficulty,
    numBoards,
    solved,
    impossible,
    totalQuadrants,
    totalCoverage,
    execution_time
)

print(f"Tempo de execução: {execution_time:.4f} segundos")