from src.BoardGeneration import *

def getCandidates(board, row , col):
    # candidates = [num for num in range(1,10) if isValid(board, row, col, num)]
    candidates = []
    for num in range(1, 10):
        if isValid(board, row, col, num):
            candidates.append(num)

    return candidates
