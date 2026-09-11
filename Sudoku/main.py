from src.Board import *
from src.BoardGeneration import *
from src.Probability import *
from src.Logs import *

import time

# ---------- Cores ANSI (opcional, funciona na maioria dos terminais) ----------
class Cor:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CIANO = "\033[96m"
    VERDE = "\033[92m"
    VERMELHO = "\033[91m"
    AMARELO = "\033[93m"
    AZUL = "\033[94m"
    CINZA = "\033[90m"

LARGURA = 50


def linha(char="-", cor=Cor.CINZA):
    print(f"{cor}{char * LARGURA}{Cor.RESET}")

def titulo(texto, cor=Cor.CIANO):
    linha("=", cor)
    print(f"{cor}{Cor.BOLD}{texto.center(LARGURA)}{Cor.RESET}")
    linha("=", cor)

def secao(texto, cor=Cor.AZUL):
    print(f"\n{cor}{Cor.BOLD}{texto}{Cor.RESET}")
    linha("-", cor)

def menu(opcoes: dict, titulo_menu: str):
    secao(titulo_menu)
    for chave, valor in opcoes.items():
        print(f"  {Cor.AMARELO}{chave}{Cor.RESET} - {valor}")
    linha("-")

# ---------------------- Configuração Inicial ----------------------
titulo("SUDOKU SOLVER")

numBoards = int(input(f"\n{Cor.BOLD}Número de jogos que deseja: {Cor.RESET}"))

menu(
    {"1": "Fácil", "2": "Médio", "3": "Difícil"},
    "DIFICULDADE"
)
difficulty = int(input(f"{Cor.BOLD}Escolha a dificuldade: {Cor.RESET}"))

menu(
    {
        "1": "Força Bruta (Probabilidade, para quando é impossível)",
        "2": "Força Bruta² (Probabilidade, só para quando ganha)",
        "3": "Usando o Cérebro (Backtracking)",
        "4":"Constraint Satisfaction Problem (CSP) Ta sendo feito"
    },
    "MÉTODO DE RESOLUÇÃO"
)
resolve = int(input(f"{Cor.BOLD}Escolha o método: {Cor.RESET}"))

limit_attempts = "Nao tem limite de Tentativas"
if resolve == 2:
    secao("LIMITE DE TENTATIVAS")
    limit_attempts = int(input(f"{Cor.AMARELO}Fale o limite de tentativas: {Cor.RESET}"))


inicio = time.time()

solved = 0
impossible = 0
totalQuadrants = 0
totalCoverage = 0
gamesPlayed = 0
totalAttempts = numBoards

try:
    for i in range(numBoards):

        titulo(f"JOGO {i + 1} / {numBoards}", Cor.AZUL)
        gamesPlayed += 1

        board = generateSudoku(difficulty)
        original = [
            [num != 0 for num in row]
            for row in board
        ]

        secao("Board Inicial")
        showBoard(board, original)

        # Escolhe o método de resolução
        if resolve == 1:
            result = solveSudoku_WithOutAttempt(board)

        elif resolve == 2:
            result, attempts = solveSudoku_WithAttempt(board, limit_attempts)
            totalAttempts += attempts

        elif resolve == 3:
            result = solveSudoku_Backtraking(board)

        else:
            print(f"\n{Cor.VERMELHO}{Cor.BOLD}Método de resolução inválido!{Cor.RESET}")
            break

        secao("Board Final")
        showBoard(board, original)

        # Calcula os quadrantes corretos
        correctQuadrants = checkQuadrants(board)
        coverage = (correctQuadrants / 9) * 100
        totalQuadrants += correctQuadrants
        totalCoverage += coverage

        # Verifica se resolveu
        if result:
            solved += 1
            status = f"{Cor.VERDE}{Cor.BOLD}✔ RESOLVIDO{Cor.RESET}"
        else:
            impossible += 1
            status = f"{Cor.VERMELHO}{Cor.BOLD}✘ IMPOSSÍVEL{Cor.RESET}"

        secao("Resultado")
        print(f"  Status:              {status}")
        print(f"  Quadrantes corretos: {Cor.AMARELO}{correctQuadrants}/9{Cor.RESET}")
        print(f"  Cobertura:           {Cor.AMARELO}{coverage:.2f}%{Cor.RESET}")
        print()


finally:

    fim = time.time()
    execution_time = fim - inicio

    titulo("RESUMO FINAL", Cor.VERDE)
    print(f"  Jogos disputados:     {gamesPlayed}")
    print(f"  Resolvidos:           {Cor.VERDE}{solved}{Cor.RESET}")
    print(f"  Impossíveis:          {Cor.VERMELHO}{impossible}{Cor.RESET}")
    print(f"  Tentativas:           {Cor.VERMELHO}{totalAttempts}{Cor.RESET}")
    if gamesPlayed:
        print(f"  Cobertura média:      {Cor.AMARELO}{totalCoverage / gamesPlayed:.2f}%{Cor.RESET}")
    print(f"  Tempo total:          {execution_time:.2f}s")
    linha("=", Cor.VERDE)

    # Salva os resultados mesmo se apertar Ctrl+C
    saveResults(
        difficulty,
        resolve,
        numBoards,
        solved,
        impossible,
        totalQuadrants,
        totalCoverage,
        limit_attempts,
        totalAttempts,
        execution_time
    )