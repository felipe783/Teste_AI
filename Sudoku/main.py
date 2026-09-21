import time
import math

from src.SamuraiGame.BoardSamuraiGeneration import *
from src.SamuraiGame.SamuraiBoard import *
from src.gameConfig import *
from src.Board import *
from src.BoardGeneration import *
from src.Probability import *
from src.Logs import *

# ---------- Cores ANSI ----------
class Cor:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CIANO = "\033[96m"
    VERDE = "\033[92m"
    VERMELHO = "\033[91m"
    AMARELO = "\033[93m"
    AZUL = "\033[94m"
    MAGENTA = "\033[95m"
    BRANCO = "\033[97m"
    CINZA = "\033[90m"

LARGURA = 50
SIZE = 33   

# ---------- Helpers de estilo ----------
def linha(char="─", cor=Cor.CINZA):
    print(f"{cor}{char * LARGURA}{Cor.RESET}")

def titulo(texto, cor=Cor.CIANO):
    interno = LARGURA - 2
    print(f"{cor}╔{'═' * interno}╗{Cor.RESET}")
    print(f"{cor}║{Cor.BOLD}{texto.center(interno)}{Cor.RESET}{cor}║{Cor.RESET}")
    print(f"{cor}╚{'═' * interno}╝{Cor.RESET}")

def secao(texto, cor=Cor.AZUL):
    print(f"\n{cor}{Cor.BOLD}▌ {texto}{Cor.RESET}")
    linha("─", cor)

def menu(opcoes: dict, titulo_menu: str):
    secao(titulo_menu)
    for chave, valor in opcoes.items():
        print(f"  {Cor.AMARELO}{Cor.BOLD}[{chave}]{Cor.RESET} {Cor.BRANCO}{valor}{Cor.RESET}")
    linha("─")

def prompt(texto, cor=Cor.BOLD):
    return f"{Cor.CIANO}{Cor.BOLD}❯{Cor.RESET} {cor}{texto}{Cor.RESET}"

def aviso(texto):
    print(f"{Cor.AMARELO}{Cor.BOLD}⚠{Cor.RESET} {Cor.AMARELO}{texto}{Cor.RESET}")

def erro(texto):
    print(f"{Cor.VERMELHO}{Cor.BOLD}✘ {texto}{Cor.RESET}")

def campo(rotulo, valor):
    pontilhado = (rotulo + " ").ljust(22, "·")
    print(f"  {Cor.CINZA}{pontilhado}{Cor.RESET} {valor}")


# MAIN FLOW

titulo("SUDOKU SOLVER")

menu(
    {"1": "Sudoku Tradicional", "2": "Sudoku Samurai (33x33)"},
    "TIPO DE JOGO"
)

gameType = int(input(prompt("Escolha o modo: ")))

if gameType == 2:
    challenge = True
else:
    secao("CONFIGURAÇÃO DO SUDOKU", Cor.MAGENTA)
    print(f"  {Cor.DIM}Ex: 9 --> 9x9, 81 --> 81x81{Cor.RESET}")
    SIZE = int(input(prompt("Digite o tamanho do Sudoku (9 para 9x9, 81 para 81x81): ")))
    if SIZE < 0:
        erro("O Número deve ser maior que 0")
        exit(0)
        
    square = math.isqrt(SIZE)

    if not(square**2 == SIZE):
        erro("O Número deve ser um Quadrado Perfeito")
        exit(0)

        

numBoards = int(input("\n" + prompt("Número de jogos que deseja: ")))


menu({"1": "Fácil", "2": "Médio", "3": "Difícil"}, "DIFICULDADE")
difficulty = int(input(prompt("Escolha a dificuldade: ")))

totalRevealedCells = setupGame(difficulty, SIZE) # Total de Celulas a Revelar
# print(difficulty, size)

# O Samurai tem uma matriz complexa que inviabiliza as lógicas de "probabilidade" pura feitas para o 9x9.
if gameType == 1:
    menu({
        "1": "Probabilidade com 1 Tentativa",
        "2": "Probabilidade com N Tentativas",
        "3": "BackTraking",
    }, "MÉTODO DE RESOLUÇÃO")
    resolve = int(input(prompt("Escolha o método: ")))
else:
    print()
    aviso("O Sudoku Samurai requer regras de Backtracking estrutural.")
    aviso("Método de Resolução definido automaticamente para: Usando o Backtracking.")
    resolve = 3

limit_attempts = "Apenas 1 tentativa"
if resolve == 2 and gameType == 1:
    secao("LIMITE DE TENTATIVAS", Cor.MAGENTA)
    limit_attempts = int(input(prompt("Fale o limite de tentativas: ", Cor.AMARELO)))
    
inicio = time.time()

solved = 0
impossible = 0
totalQuadrants = 0
totalCoverage = 0
gamesPlayed = 0
totalAttempts = numBoards

try:
    for i in range(numBoards):

        print()
        titulo(f"JOGO {i + 1} / {numBoards}", Cor.AZUL)
        gamesPlayed += 1

        if gameType == 1:
            # Fluxo Tradicional 9x9
            board = generateSudoku(totalRevealedCells, SIZE)
            original = [[num != 0 for num in row] for row in board]
            
            secao("Board Inicial")
            showBoard(board, original, SIZE)

            if resolve == 1:
                result = solveSudoku_WithOutAttempt(board, SIZE)
            elif resolve == 2:
                result, attempts = solveSudoku_WithAttempt(board, limit_attempts, SIZE)
                totalAttempts += attempts
            elif resolve == 3:
                result = solveSudoku_Backtraking(board, SIZE)
            elif resolve == 4:
                pass # Substitua pela sua chamada CSP real
                
            secao("Board Final", Cor.VERDE)
            showBoard(board, original, SIZE)

            correctQuadrants = checkQuadrants(board,SIZE)
            coverage = (correctQuadrants / 9) * 100
            totalQuadrants += correctQuadrants
            totalCoverage += coverage

        else:
            # Fluxo Samurai 33x33
            board = generateSamurai(totalRevealedCells)
            # Para renderização: mapeia células válidas e que não estão vazias como Originais
            original = [[(num != 0 and num is not None) for num in row] for row in board]
            
            secao("Board Inicial (Samurai)")
            showBoardSamurai(board, original)

            result = solveSamurai_Backtracking(board)

            secao("Board Final (Samurai)", Cor.VERDE)
            showBoardSamurai(board, original)

            # Cálculo de Cobertura para o Samurai
            celulas_preenchidas = 0
            celulas_totais = 0
            for r in range(SIZE):
                for c in range(SIZE):
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

        secao("Resultado", Cor.MAGENTA)
        campo("Status", status)
        if gameType == 1:
            campo("Quadrantes corretos", f"{Cor.AMARELO}{Cor.BOLD}{correctQuadrants}/9{Cor.RESET}")
        campo("Cobertura", f"{Cor.AMARELO}{Cor.BOLD}{coverage:.2f}%{Cor.RESET}")
        print()

finally:
    fim = time.time()
    execution_time = fim - inicio

    print()
    titulo("RESUMO FINAL", Cor.VERDE)
    campo("Jogos disputados", f"{Cor.BRANCO}{Cor.BOLD}{gamesPlayed}{Cor.RESET}")
    campo("Resolvidos", f"{Cor.VERDE}{Cor.BOLD}{solved}{Cor.RESET}")
    campo("Impossíveis", f"{Cor.VERMELHO}{Cor.BOLD}{impossible}{Cor.RESET}")
    
    if gameType == 1:
        campo("Tentativas", f"{Cor.VERMELHO}{Cor.BOLD}{totalAttempts}{Cor.RESET}")
        
    if gamesPlayed:
        campo("Cobertura média", f"{Cor.AMARELO}{Cor.BOLD}{totalCoverage / gamesPlayed:.2f}%{Cor.RESET}")
        
    campo("Tempo total", f"{Cor.CIANO}{Cor.BOLD}{execution_time:.2f}s{Cor.RESET}")
    linha("═", Cor.VERDE)

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
        SIZE,
        gameType
    )