from src.BoardGeneration import *
from src.Board import *
import random

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

def getNumber(candidates, weights):
    total = sum(weights)
    value = random.uniform(0, total) # Sorteador um Decimal entre 0 e total(soma dos pesos)
    accumulated = 0

    for i, weight in enumerate(weights):
        accumulated += weight
        if value < accumulated:
            return candidates[i]

def checkQuadrants(board):
    correct = 0

    for start_row in range(0, 9, 3):
        for start_col in range(0, 9, 3):

            numbers = []

            for row in range(start_row, start_row + 3):
                for col in range(start_col, start_col + 3):
                    numbers.append(board[row][col])

            if sorted(numbers) == list(range(1, 10)):
                correct += 1

    return correct

def solveAttempt(board):

    while not checkVictory(board):

        row, col = findEmpty(board)
        candidates = getCandidates(board, row, col)

        if not candidates:
            return False

        weights = getWeights(board, candidates)
        # number = random.choices(candidates,weights=weights,k=1)[0] # Esta linha sortea um elemento usando os Pesos, e pega o elemento da posição 0, o K é quantos numeros queremos retornar(no caso 1)
        number = getNumber(candidates, weights)
        board[row][col] = number

    return True

def solveProbabilistic(board):  

    attempts = 0
    original = [row[:] for row in board]

    while not checkVictory(board):
        attempts += 1
        for row in range(9):
            for col in range(9):
                board[row][col] = original[row][col]
                
        result = solveAttempt(board)

        if result:
            print(f"Solução encontrada em {attempts} tentativas")
            return board

    return board