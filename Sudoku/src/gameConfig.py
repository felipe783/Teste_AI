def setupGame():
    print("=== CONFIGURAÇÃO DO SUDOKU ===")

    while True:
        try:
            input = input("Digite o tamanho do Sudoku (9 para 9x9, 33 para Samurai): ")

            if "x" in input:
                user_input = user_input.lower().split("x")[0].strip()
                break
            if not (input.lower().split("x")[0].strip() == input.lower().split("x")[0].strip()):
                raise ValueError("Entrada inválida. Por favor, insira um quadrado perfeito.")
            
        except ValueError as e:
            print(e)

    totalCells = int(user_input) ** 2
    
    # Dicionário contendo a dificuldade e a % de casas que vão FICAR reveladas
    difficultyRatios = {
        1: ("Fácil", totalCells * 0.50),   # 50% das casas reveladas
        2: ("Médio", totalCells * 0.40),   # 40% das casas reveladas
        3: ("Difícil", totalCells * 0.30)  # 30% das casas reveladas
    }

    return difficultyRatios, user_input

