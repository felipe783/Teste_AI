import time
import copy
import random
import core.config as core

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
GRID_SIZE = 33

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


# =====================================================================
#                      LÓGICA DO SUDOKU SAMURAI
# =====================================================================

def createGridSamurai():
    grid = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for _, (row, col) in core.SAMURAI_CONFIG:
        for i in range(9):
            for j in range(9):
                grid[row + i][col + j] = 0
    return grid

def isValidSamuraiSub(board, row, col, num, startRow, startCol):
    # Linha
    for c in range(startCol, startCol + 9):
        if board[row][c] == num:
            return False
    # Coluna
    for r in range(startRow, startRow + 9):
        if board[r][col] == num:
            return False
    # Bloco 3x3
    localRow, localCol = row - startRow, col - startCol
    blockRow, blockCol = startRow + (localRow // 3) * 3, startCol + (localCol // 3) * 3
    for r in range(blockRow, blockRow + 3):
        for c in range(blockCol, blockCol + 3):
            if board[r][c] == num:
                return False
    return True

def isValidSamuraiGlobally(board, row, col, num):
    for _, (startRow, startCol) in core.SAMURAI_CONFIG:
        if (startRow <= row < startRow + 9 and startCol <= col < startCol + 9):
            if not isValidSamuraiSub(board, row, col, num, startRow, startCol):
                return False
    return True

def findEmptySamurai(board):
    minOptions = 10
    bestCell = None
    bestValidNumbers = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if board[row][col] == 0:
                validNumbers = [num for num in range(1, 10) if isValidSamuraiGlobally(board, row, col, num)]
                numOptions = len(validNumbers)

                if numOptions == 0:
                    return row, col, []
                if numOptions < minOptions:
                    minOptions = numOptions
                    bestCell = (row, col)
                    bestValidNumbers = validNumbers
                    if minOptions == 1:
                        return bestCell[0], bestCell[1], bestValidNumbers
                        
    if bestCell:
        return bestCell[0], bestCell[1], bestValidNumbers
    return None, None, None

def solveSamurai_Backtracking(board):
    row, col, validNums = findEmptySamurai(board)
    if row is None:
        return True
    if not validNums:
        return False
        
    random.shuffle(validNums)
    for num in validNums:
        board[row][col] = num
        if solveSamurai_Backtracking(board):
            return True
        board[row][col] = 0
    return False

def removeNumbersSamurai(board, amount):
    positions = []
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] != 0 and board[row][col] is not None:
                positions.append((row, col))
                
    random.shuffle(positions) 
    currentNumbers = len(positions)

    for row, col in positions:
        if currentNumbers <= amount: 
            break
        board[row][col] = 0
        currentNumbers -= 1
    return board

def generateSamurai(difficulty):
    print(f"\n{Cor.CINZA}Gerando tabuleiro Samurai (pode levar alguns segundos)...{Cor.RESET}")
    grid = createGridSamurai()
    
    # Mapeando dificuldade para a quantidade de números que sobram (exemplo)
    if difficulty == 1: amount = 300
    elif difficulty == 2: amount = 200
    else: amount = getattr(core, 'SAMURAI_NUMBERS_HARD', 120)

    if solveSamurai_Backtracking(grid):
        puzzleGrid = removeNumbersSamurai(grid, amount)
        return puzzleGrid
    return grid

# --- ALGORITMO EXCLUSIVO DE EXIBIÇÃO DO SAMURAI ---
def showBoardSamurai(board, original):
    print()
    for r in range(len(board)):
        linha_str = ""
        tem_conteudo_na_linha = False
        
        for c in range(len(board[r])):
            val = board[r][c]
            is_orig = original[r][c]

            if val is not None:
                tem_conteudo_na_linha = True

            # Formatação do caractere e cor
            if val is None:
                char = " "  # Célula fora do grid de jogo
            elif val == 0:
                char = f"{Cor.CINZA}.{Cor.RESET}"
            elif is_orig:
                char = f"{Cor.CIANO}{Cor.BOLD}{val}{Cor.RESET}"
            else:
                char = f"{Cor.VERDE}{val}{Cor.RESET}"

            linha_str += char + " "

            # Espaçamento vertical para delimitar blocos 3x3 de forma agradável
            if c % 3 == 2 and val is not None:
                linha_str += " "
            elif c % 3 == 2 and val is None:
                linha_str += " "

        if tem_conteudo_na_linha:
            print(linha_str)
            # Espaçamento horizontal para delimitar blocos 3x3
            if r % 3 == 2:
                print()


# =====================================================================
#                              MAIN FLOW
# =====================================================================

titulo("SUDOKU SOLVER")

menu(
    {"1": "Sudoku Tradicional (9x9)", "2": "Sudoku Samurai (33x33)"},
    "TIPO DE JOGO"
)
tipo_jogo = int(input(f"{Cor.BOLD}Escolha o modo: {Cor.RESET}"))

numBoards = int(input(f"\n{Cor.BOLD}Número de jogos que deseja: {Cor.RESET}"))

menu({"1": "Fácil", "2": "Médio", "3": "Difícil"}, "DIFICULDADE")
difficulty = int(input(f"{Cor.BOLD}Escolha a dificuldade: {Cor.RESET}"))

# O Samurai tem uma matriz complexa que inviabiliza as lógicas de "probabilidade" pura feitas para o 9x9.
if tipo_jogo == 1:
    menu({
        "1": "Força Bruta (Probabilidade, impossível)",
        "2": "Força Bruta² (Probabilidade, vitória)",
        "3": "Usando o Cérebro (Backtracking)",
        "4": "CSP (Constraint Satisfaction Problem)"
    }, "MÉTODO DE RESOLUÇÃO")
    resolve = int(input(f"{Cor.BOLD}Escolha o método: {Cor.RESET}"))
else:
    print(f"\n{Cor.AMARELO}{Cor.BOLD}[!] O Sudoku Samurai requer regras de Backtracking estrutural.{Cor.RESET}")
    print(f"{Cor.AMARELO}[!] Método de Resolução definido automaticamente para: Usando o Cérebro (Backtracking).{Cor.RESET}")
    resolve = 3

limit_attempts = "Nao tem limite de Tentativas"
if resolve == 2 and tipo_jogo == 1:
    secao("LIMITE DE TENTATIVAS")
    limit_attempts = int(input(f"{Cor.AMARELO}Fale o limite de tentativas: {Cor.RESET}"))
    
challenge = 0
if resolve == 4 and tipo_jogo == 1:
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

        if tipo_jogo == 1:
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
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
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
        if tipo_jogo == 1:
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
    
    if tipo_jogo == 1:
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
        execution_time
    )