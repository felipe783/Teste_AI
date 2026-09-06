from src.Board import *
from src.BoardGeneration import *
from src.Probability import *
import time
from Core.BoardPatterns import boardPatterns

numBoards = 10
inicio = time.time()

for i in range(numBoards): # Variavel descartável 
    print(f"Jogo {i + 1}")
    board = generateSudoku("easy")

    print("Board Inicial:")
    showBoard(board)

    solveProbabilistic(board)

    print("Board Final")
    showBoard(board)

fim = time.time()
print(f"Tempo de execução: {fim - inicio:.4f} segundos")
'''
difficulties = ["Easy", "Medium", "Hard", "Victorious Board"]
for i, board in enumerate(boardPatterns): # O Enumarete pega já o Indice e o Board, é basicamente um "board[i]" so que mais limpo
    print(f"\n--- {difficulties[i]} ---")
    showBoard(board)
    print("Vitoria!" if checkVictory(board) else "Muito Ruim!")
'''