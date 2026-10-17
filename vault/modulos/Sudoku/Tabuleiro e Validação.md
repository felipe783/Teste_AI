# Sobre o Arquivo

O arquivo `Board.py` é o **Responsável pela Exibição e pela Verificação Final** do tabuleiro tradicional.
# Funcionalidades

## _blockDims

Encontra a divisão `(linhas, colunas)` do bloco mais próxima de um quadrado, para que a exibição funcione com qualquer tamanho: `9 → (3,3)`, `6 → (2,3)`, `12 → (3,4)`, `16 → (4,4)`.

## showBoard

Desenha o tabuleiro com bordas Unicode e cores ANSI:
- **cinza** `.` → célula vazia;
- **verde** → número original (revelado);
- **azul** → número preenchido pelo algoritmo.

A largura da célula se ajusta ao maior número (`len(str(size))`), então 16x16 e 81x81 ficam alinhados.

## checkVictory

Verifica se cada **linha**, **coluna** e **bloco** contém exatamente `1..size`, usando `sorted(numbers) == list(range(1, size + 1))`.

````python
for row_start in range(0, size, square):
    for column_start in range(0, size, square):
        numbers = []
        for row in range(row_start, row_start + square):
            for column in range(column_start, column_start + square):
                numbers.append(board[row][column])
        if sorted(numbers) != list(range(1, size + 1)):
            return False
````

> [!note] Duas noções de bloco
> `showBoard` usa `_blockDims` (aceita 6, 12…), mas `checkVictory`, `isValid` e `checkQuadrants` usam `isqrt(size)`. Isso só é consistente porque o `main.py` aceita apenas quadrados perfeitos.
