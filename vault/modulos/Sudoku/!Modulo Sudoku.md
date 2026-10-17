O módulo **`Sudoku`** é um **gerador e resolvedor de Sudoku** escrito só com a biblioteca padrão do Python. Ele compara estratégias de resolução: probabilidade ponderada, tentativas repetidas e Backtracking. Suporta tabuleiros NxN (N quadrado perfeito) e o Sudoku Samurai.

Organização:
- **`main.py`**
    - Menus, fluxo da bateria de jogos e métricas. Ver [[Main Sudoku]].
- **`src/`**
    - Regras, geração e resolução do Sudoku tradicional: [[Geração do Tabuleiro]], [[Resolvedores]], [[Tabuleiro e Validação]], [[Logs Sudoku]].
- **`src/SamuraiGame/`**
    - Estrutura própria para a matriz 33x33. Ver [[Sudoku Samurai]].
- **`core/`**
    - Configuração das posições dos tabuleiros do Samurai (`config.py`).

> [!note] Imports
> Os arquivos usam `from src.X import *` e `import core.config`, então o programa precisa ser executado **de dentro da pasta `Sudoku/`** (`python main.py`).
