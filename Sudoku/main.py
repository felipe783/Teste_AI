import time
import core.config as core

from src.SamuraiGame.BoardSamuraiGeneration import *
from src.SamuraiGame.SamuraiBoard import *
from src.Board import *
from src.BoardGeneration import *
from src.Probability import *
from src.Logs import *

# ---------- Cores ANSI ----------
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


# MAIN FLOW

titulo("SUDOKU SOLVER")

menu(
    {"1": "Sudoku Tradicional (9x9)", "2": "Sudoku Samurai (33x33)"},
    "TIPO DE JOGO"
)

gameType = int(input(f"{Cor.BOLD}Escolha o modo: {Cor.RESET}"))

if gameType == 2:
    challenge = True
numBoards = int(input(f"\n{Cor.BOLD}Número de jogos que deseja: {Cor.RESET}"))

menu({"1": "Fácil", "2": "Médio", "3": "Difícil"}, "DIFICULDADE")
difficulty = int(input(f"{Cor.BOLD}Escolha a dificuldade: {Cor.RESET}"))

# O Samurai tem uma matriz complexa que inviabiliza as lógicas de "probabilidade" pura feitas para o 9x9.
if gameType == 1:
    menu({
        "1": "Probabilidade com 1 Tentativa",
        "2": "Probabilidade com N Tentativas",
        "3": "BackTraking",
    }, "MÉTODO DE RESOLUÇÃO")
    resolve = int(input(f"{Cor.BOLD}Escolha o método: {Cor.RESET}"))
else:
    print(f"\n{Cor.AMARELO}{Cor.BOLD}[!] O Sudoku Samurai requer regras de Backtracking estrutural.{Cor.RESET}")
    print(f"{Cor.AMARELO}[!] Método de Resolução definido automaticamente para: Usando o Backtracking.{Cor.RESET}")
    resolve = 3

limit_attempts = "Nao tem limite de Tentativas"
if resolve == 2 and gameType == 1:
    secao("LIMITE DE TENTATIVAS")
    limit_attempts = int(input(f"{Cor.AMARELO}Fale o limite de tentativas: {Cor.RESET}"))
    
challenge = 0
if resolve == 4 and gameType == 1:
    menu({"1": "SIM", "2": "NAO"}, "DESEJA O DESAFIO?")
    challenge = int(input(f"{Cor.AMARELO}Fale: {Cor.RESET}"))
    if challenge == 1:
        secao(" DESAFIO ")
    else:
        challenge = 0

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

        if gameType == 1:
            # Fluxo Tradicional 9x9
            board = generateSudoku(difficulty)
            original = [[num != 0 for num in row] for row in board]
            
            secao("Board Inicial")
            showBoard(board, original)

            if resolve == 1:
                result = solveSudoku_WithOutAttempt(board)
            elif resolve == 2:
                result, attempts = solveSudoku_WithAttempt(board, limit_attempts)
                totalAttempts += attempts
            elif resolve == 3:
                result = solveSudoku_Backtraking(board)
            elif resolve == 4:
                pass # Substitua pela sua chamada CSP real
                
            secao("Board Final")
            showBoard(board, original)

            correctQuadrants = checkQuadrants(board)
            coverage = (correctQuadrants / 9) * 100
            totalQuadrants += correctQuadrants
            totalCoverage += coverage

        else:
            # Fluxo Samurai 33x33
            board = generateSamurai(difficulty)
            # Para renderização: mapeia células válidas e que não estão vazias como Originais
            original = [[(num != 0 and num is not None) for num in row] for row in board]
            
            secao("Board Inicial (Samurai)")
            showBoardSamurai(board, original)

            result = solveSamurai_Backtracking(board)

            secao("Board Final (Samurai)")
            showBoardSamurai(board, original)

            # Cálculo de Cobertura para o Samurai
            celulas_preenchidas = 0
            celulas_totais = 0
            for r in range(core.GRID_SIZE):
                for c in range(core.GRID_SIZE):
                    if board[r][c] is not None:
                        celulas_totais += 1
                        if board[r][c] != 0:
                            celulas_preenchidas += 1
                            
            coverage = (celulas_preenchidas / celulas_totais) * 100 if celulas_totais > 0 else 0
            totalCoverage += coverage
            totalQuadrants += 9 # Apenas para fins de visualização nos logs finais

        # Status de Finalização
        if result:
            solved += 1
            status = f"{Cor.VERDE}{Cor.BOLD}✔ RESOLVIDO{Cor.RESET}"
        else:
            impossible += 1
            status = f"{Cor.VERMELHO}{Cor.BOLD}✘ IMPOSSÍVEL{Cor.RESET}"

        secao("Resultado")
        print(f"  Status:              {status}")
        if gameType == 1:
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
    
    if gameType == 1:
        print(f"  Tentativas:           {Cor.VERMELHO}{totalAttempts}{Cor.RESET}")
        
    if gamesPlayed:
        print(f"  Cobertura média:      {Cor.AMARELO}{totalCoverage / gamesPlayed:.2f}%{Cor.RESET}")
        
    print(f"  Tempo total:          {execution_time:.2f}s")
    linha("=", Cor.VERDE)

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
        execution_time,
        
        gameType
    )