# Sobre os Arquivos

Os arquivos `gameConfig.py` e `BoardGeneration.py` são **Responsáveis por Criar o Puzzle**: definem quantas células ficam visíveis e geram um tabuleiro completo e válido, do qual são removidos números.
# setupGame

Responsabilidade:
- Converter a dificuldade em **quantidade de células reveladas**.

````python
def setupGame(difficulty, size):
    totalCells = int(size) ** 2
    if difficulty == 1:   totalRevealedCells = int(totalCells * 0.50)  # Fácil
    elif difficulty == 2: totalRevealedCells = int(totalCells * 0.40)  # Médio
    elif difficulty == 3: totalRevealedCells = int(totalCells * 0.30)  # Difícil
````

> [!note] Nome enganoso
> O valor é passado para `generateSudoku(removeCells, size)` e depois para `removerNumbers(board, amount, size)` como `amount`, mas significa **quantas células manter**, não quantas remover.

# Funcionalidades — BoardGeneration

## isValid

Verifica se um número **pode ser colocado** numa célula: não pode repetir na linha, na coluna nem no bloco `√size × √size`.
- O início do bloco é `(row // square) * square`. Ex.: linha 7 num 9x9 → `7 // 3 = 2` → começa na linha 6.

## findEmpty (heurística MRV)

Percorre as células vazias e calcula os candidatos válidos de cada uma, guardando a célula com **menos candidatos** (*Minimum Remaining Values*):
- Se alguma célula tem **0 candidatos**, retorna `(row, col, [])` (tabuleiro impossível).
- Se encontra uma com **1 candidato**, retorna imediatamente.
- Se não há vazias, retorna `None`.

````python
if not candidates:
    return row, col, []            # Sudoku impossivel
if len(candidates) < minOptions:
    minOptions = len(candidates)
    bestCell = (row, col)
    bestCandidates = candidates
    if minOptions == 1:
        return row, col, candidates
````

## generateSolution

Backtracking **randomizado**: embaralha os candidatos da célula MRV e tenta cada um recursivamente. Como parte de um tabuleiro vazio, gera uma solução completa diferente a cada execução.

````python
random.shuffle(candidates)
for num in candidates:
    board[row][col] = num
    if generateSolution(board, size):
        return True
    board[row][col] = 0
return False
````

## removerNumbers

Coleta as posições preenchidas, embaralha e zera células até restarem `amount` números.

> [!note] Gerador alternativo comentado
> No fim do arquivo há, dentro de uma string `"""`, um gerador por **Quadrado Latino** (`pattern(r, c)` com linhas/colunas embaralhadas) e um `isValidSolution`. Ele não é usado no fluxo atual.
