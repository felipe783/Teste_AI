# Sobre o Arquivo

O arquivo `main.py` é o **Ponto de Entrada do Sudoku**: monta os menus coloridos no terminal, coleta as escolhas do usuário, executa a bateria de jogos e exibe o resumo final.

Responsabilidades:
- Definir as cores ANSI (`class Cor`) e os helpers de estilo (`titulo`, `secao`, `menu`, `campo`)
- Ler tipo de jogo, tamanho, número de jogos, dificuldade e método
- Executar cada jogo e calcular as métricas
- Chamar [[Logs Sudoku|saveResults]] no `finally`
# Funcionalidades

## Menus e Validação de Entrada

- **Tipo**: `1` tradicional ou `2` Samurai.
- **Tamanho** (só tradicional): precisa ser quadrado perfeito, verificado com `math.isqrt(SIZE) ** 2 == SIZE`.
- **Dificuldade**: `1` Fácil, `2` Médio, `3` Difícil → [[Geração do Tabuleiro#setupGame|setupGame]].
- **Método**: `1` Probabilidade com 1 tentativa, `2` com N tentativas, `3` Backtracking. No Samurai o método é sempre `3`.

````python
square = math.isqrt(SIZE)
if not(square**2 == SIZE):
    erro("O Número deve ser um Quadrado Perfeito")
    exit(0)
````

## Execução de cada Jogo

Para o tradicional, gera o tabuleiro, guarda a máscara `original` (células reveladas), chama o [[Resolvedores|resolvedor]] e conta os quadrantes corretos:

````python
board = generateSudoku(totalRevealedCells, SIZE)
original = [[num != 0 for num in row] for row in board]

if resolve == 1:
    result = solveSudoku_WithOutAttempt(board, SIZE)
elif resolve == 2:
    result, attempts, _ = solveSudoku_WithAttempt(board, limit_attempts, SIZE)
elif resolve == 3:
    result = solveSudoku_Backtraking(board, SIZE)
````

Para o Samurai, a cobertura é a fração de células válidas (`!= None`) preenchidas.

## Resumo Final

O bloco `finally` garante que o resumo seja exibido e salvo mesmo que a execução seja interrompida no meio (por exemplo, com `Ctrl+C`).

> [!warning] Veredito "resolvido" incorreto (confirmado em 2026-10-17)
> - O método 1 retorna `(board, bool)` e o método 3 retorna `(board, True)` ou `False`. O `if result:` testa a tupla, que é sempre verdadeira. Na execução, o método 1 marcou "✔ RESOLVIDO" com 3/9 quadrantes.
> - O método 2 descarta o booleano (`result, attempts, _`) e `result` é o próprio `board`, uma lista não vazia, portanto também sempre "resolvido".
> - A cobertura usa `correctQuadrants / 9` fixo, o que só está certo para 9x9.
> - A opção `4` (CSP) existe no código como `pass`, mas não aparece no menu.
