from board import *
from Core.BoardPatterns import boardPatterns

difficulties = ["Easy", "Medium", "Hard", "Victorious Board"]

for i, board in enumerate(boardPatterns): # O Enumarete pega já o Indice e o Board, é basicamente um "board[i]" so que mais limpo
    print(f"\n--- {difficulties[i]} ---")
    showBoard(board)
    print("Vitoria!" if checkVictory(board) else "Muito Ruim!")