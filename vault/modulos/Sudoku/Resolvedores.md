# Sobre o Arquivo

O arquivo `Probability.py` concentra os **Algoritmos de Resolução** do Sudoku tradicional e o cálculo de quadrantes corretos.
# Probabilidade Ponderada

## getCandidates, checkNumberDuplicates e getWeights

- `getCandidates`: números válidos para a célula, segundo [[Geração do Tabuleiro|isValid]].
- `checkNumberDuplicates`: para cada candidato, conta **em quantas células vazias do tabuleiro** ele ainda caberia.
- `getWeights`: peso **inversamente proporcional** a essa contagem, `1 / (count + 1)`. Um número que cabe em poucos lugares é mais "urgente" e recebe peso maior.

## getNumber (roleta)

Sorteia um valor em `[0, soma dos pesos)` e devolve o candidato cujo intervalo acumulado contém o sorteio. É equivalente a `random.choices(candidates, weights)`, que aparece comentado.

````python
value = random.uniform(0, total)
accumulated = 0
for i, weight in enumerate(weights):
    accumulated += weight
    if value < accumulated:
        return candidates[i]
````

# Métodos

| Método (menu) | Função | Ideia | Retorno |
|---|---|---|---|
| 1 — 1 tentativa | `solveSudoku_WithOutAttempt` | Preenche a célula MRV com um número sorteado por peso; não volta atrás | `(board, bool)` |
| 2 — N tentativas | `solveSudoku_WithAttempt` + `solveAttempt` | Restaura o tabuleiro original e repete a tentativa até vencer ou atingir o limite | `(board, attempts, bool)` |
| 3 — Backtracking | `solveSudoku_Backtraking` | DFS com MRV: testa candidatos e retrocede ao encontrar conflito | `(board, True)` ou `False` |

## Backtracking

````python
empty = findEmpty(board, size)
if empty is None:
    return board, True
row, col, candidates = empty
while candidates:
    number = candidates.pop()
    board[row][col] = number
    if solveSudoku_Backtraking(board, size):
        return board, True
    board[row][col] = 0
return False
````

> [!warning] Tipos de retorno inconsistentes
> Os três métodos retornam formatos diferentes, e o [[Main Sudoku]] trata todos como booleano. Por isso o veredito "resolvido" está errado nos métodos 1 e 2.

## checkQuadrants

Conta quantos blocos `√size × √size` contêm exatamente os números `1..size`. É a métrica de "quadrantes corretos".

> [!note] Recursão
> O `Backtracking` é recursivo, por isso `main.py` e `Probability.py` chamam `sys.setrecursionlimit(1000000)`. Em tabuleiros grandes (ex.: 81x81) o tempo cresce muito, e o limite real passa a ser a pilha do sistema.
