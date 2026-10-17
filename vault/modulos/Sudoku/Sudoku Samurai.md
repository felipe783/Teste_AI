# Sobre os Arquivos

Os arquivos `core/config.py`, `src/SamuraiGame/BoardSamuraiGeneration.py` e `src/SamuraiGame/SamuraiBoard.py` implementam o **Sudoku Samurai**: vários tabuleiros 9x9 numa matriz global 33x33, com regiões compartilhadas.
# Configuração

````python
SAMURAI_CONFIG = [
    ("SUDOKU_A", (0, 12)),   # Linha inicial, coluna inicial
    ("SUDOKU_B", (6, 18)),
    ("SUDOKU_C", (6, 6)),
    ("SUDOKU_D", (12, 0)),
    ("SUDOKU_E", (12, 12)),
    ("SUDOKU_F", (12, 24)),
    ("SUDOKU_G", (18, 6)),
    ("SUDOKU_H", (18, 18)),
    ("SUDOKU_I", (24, 12)),
]
GRID_SIZE = 33
````

- São **9 tabuleiros**, não os 5 do Samurai clássico. A união deles ocupa **621** das 1089 posições da matriz.
- As posições fora de qualquer tabuleiro valem `None`; as células jogáveis começam em `0`.

# Funcionalidades

## Validação em dois níveis

- `isValidSamuraiSub`: valida linha, coluna e bloco 3x3 **dentro de um tabuleiro**, convertendo a posição global em local (`local = global − início do Sudoku`).
- `isValidSamuraiGlobally`: aplica a validação em **todos os tabuleiros que contêm a célula**, garantindo as regras nas regiões sobrepostas.

````python
for _, (startRow, startCol) in core.SAMURAI_CONFIG:
    if (startRow <= row < startRow + 9 and startCol <= col < startCol + 9):
        if not isValidSamuraiSub(board, row, col, num, startRow, startCol):
            return False
return True
````

## Geração e Resolução

- `findEmptySamurai`: MRV sobre a matriz global (ignora `None`).
- `solveSamurai_Backtracking`: Backtracking randomizado. Gera a solução a partir da grade vazia e também resolve o puzzle depois.
- `generateSamurai(removeCells)`: cria a grade, resolve e chama `removeNumbersSamurai` para manter `removeCells` números.

## Exibição

`showBoardSamurai` imprime a matriz omitindo linhas sem conteúdo e separando blocos 3x3 com espaços extras: ciano para originais e verde para preenchidos.

> [!warning] Dificuldade calculada sobre a matriz inteira (confirmado em 2026-10-17)
> O `main.py` chama `setupGame(difficulty, 33)`, que usa `33² = 1089` células. Como a grade real tem 621, ficam reveladas 544 (88%), 435 (70%) e 326 (52%), em vez de 50%, 40% e 30%.
