from src.BoardGeneration import *
from src.Board import *
import random

def getCandidates(board, row , col, size):
    # candidates = [num for num in range(1,10) if isValid(board, row, col, num)]
    candidates = []
    for num in range(1, size + 1):
        if isValid(board, row, col, num, size):
            candidates.append(num)

    return candidates

def checkNumberDuplicates(board, candidates,size): # Ver quantas vezes o Número se repete  
    times = []

    for candidate in candidates:
        count = 0  

        for row in range(size):
            for col in range(size):
                if board[row][col] == 0:
                    if isValid(board, row, col, candidate, size):
                        count += 1
        times.append(count)

    return times

def getWeights(board, candidates, size):
    duplicates = checkNumberDuplicates(board, candidates, size)
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

def checkQuadrants(board, size):
    correct = 0
    square = math.isqrt(size)

    for start_row in range(0, size, square):
        for start_col in range(0, size, square):

            numbers = []

            for row in range(start_row, start_row + square):
                for col in range(start_col, start_col + square):
                    numbers.append(board[row][col])

            if sorted(numbers) == list(range(1, size + 1)):
                correct += 1

    return correct

def solveAttempt(board, size):

    while not checkVictory(board):

        row, col, _ = findEmpty(board, size)
        candidates = getCandidates(board, row, col)

        if not candidates:
            return False

        weights = getWeights(board, candidates, size)
        # number = random.choices(candidates,weights=weights,k=1)[0] # Esta linha sortea um elemento usando os Pesos, e pega o elemento da posição 0, o K é quantos numeros queremos retornar(no caso 1)
        number = getNumber(candidates, weights)
        board[row][col] = number

    return True

def solveSudoku_WithAttempt(board,limit_attempts,size):  

    attempts = 0
    original = [row[:] for row in board]

    while not checkVictory(board):
        attempts += 1
        # print(attempts)
        for row in range(size):
            for col in range(size):
                board[row][col] = original[row][col]
                
        result = solveAttempt(board)

        if result:
            print(f"Solução encontrada em {attempts} tentativas")
            return board, attempts, True
        if attempts == limit_attempts:
            print(f"Não foi possivel uma Solução em menos de {attempts} tentativas")
            return board, attempts, False

    return board, attempts, False

def solveSudoku_WithOutAttempt(board, size):  

    while not checkVictory(board, size):
        result = findEmpty(board, size)

        if result is None:
            return board, checkVictory(board, size)
        
        row, col, candidates = result

        if not candidates:
            return board, False

        weights = getWeights(board, candidates, size)
        # number = random.choices(candidates,weights=weights,k=1)[0] # Esta linha sortea um elemento usando os Pesos, e pega o elemento da posição 0, o K é quantos numeros queremos retornar(no caso 1)
        number = getNumber(candidates, weights)
        board[row][col] = number

    return board, True

calls = 0
def solveSudoku_Backtraking(board, size):
    global calls
    calls += 1
    if calls % 1000 == 0:
        print(calls, "chamadas | vazias:", sum(row.count(0) for row in board))

    empty = findEmpty(board, size)

    if empty is None:
        return board, True

    row, col, candidates = empty
    weights = getWeights(board, candidates, size)

    while candidates:

        number = getNumber(candidates, weights)
        index = candidates.index(number)

        candidates.pop(index)
        weights.pop(index)

        board[row][col] = number

        if solveSudoku_Backtraking(board, size):
            return board, True

        board[row][col] = 0

    return False