from src.board import *
from src.BoardGeneration import *
import time
from Core.BoardPatterns import boardPatterns

numBoards = 1
inicio = time.time()

for _ in range(numBoards): # Variavel descartável 
    board = generateSudoku("hard")
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