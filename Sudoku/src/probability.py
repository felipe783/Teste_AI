from src.BoardGeneration import *

def getCandidates(board, row , col):
    # candidates = [num for num in range(1,10) if isValid(board, row, col, num)]
    candidates = []
    for num in range(1, 10):
        if isValid(board, row, col, num):
            candidates.append(num)

    return candidates

def checkNumberDuplicates(board, candidates): # Ver quantas vezes o Número se repete  
    times = []

    for candidate in candidates:
        count = 0  

        for row in range(9):
            for col in range(9):

                if board[row][col] == 0:
                    if isValid(board, row, col, candidate):
                        count += 1

        times.append(count)

    return times

def getWeights(board, candidates):
    duplicates = checkNumberDuplicates(board, candidates)
    weights = []

    for duplicate in duplicates:
        weights.append(1 / (duplicate + 1)) # Peso Inversamente proporcional, quanto + vezes aparece menor é
 
    return weights

def getProbability(weights):
    total = sum(weights)
    probability = []

    for weight in weights:
        probability.append(weight / total)

    return probability