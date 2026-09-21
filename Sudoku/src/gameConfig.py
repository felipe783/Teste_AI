def setupGame(difficulty,size):
    totalCells = int(size) ** 2


    # print(totalCells)
    if difficulty == 1: # Facil
        totalRevealedCells =  int(totalCells * 0.50)
    elif difficulty == 2: # Médio
        totalRevealedCells =  int(totalCells * 0.40)
    elif difficulty == 3: # Dificil
        totalRevealedCells =  int(totalCells * 0.30)
    else:
        totalRevealedCells =  int(totalCells * 0.50)
    # print(totalRevealedCells)
    return totalRevealedCells