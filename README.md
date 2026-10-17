# Teste_AI

Repositório de experimentos com algoritmos de IA aplicados a jogos. Tem dois subprojetos independentes:

| Subprojeto | O que faz | Tecnologias |
|---|---|---|
| [SnakeGame](SnakeGame/README.md) | Uma IA aprende a jogar Snake por **Aprendizado por Reforço (DQN)** e por **Algoritmo Genético**, com treino local ou em servidor Docker | Python, PyTorch, pygame, OpenCV, Docker |
| [Sudoku](Sudoku/README.md) | Gera e resolve Sudoku tradicional (NxN) e Samurai com **probabilidade ponderada** e **Backtracking**, comparando os métodos | Python (só biblioteca padrão) |

O ambiente do Snake foi adaptado do repositório [vedantgoswami/SnakeGameAI](https://github.com/vedantgoswami/SnakeGameAI).

## 🚀 Como rodar

Requisitos: Python 3.12.

```bash
git clone https://github.com/felipe783/Teste_AI.git
cd Teste_AI

python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r SnakeGame/requirements.txt
```

Depois, siga o README do subprojeto:

```bash
cd Sudoku && python main.py                # Sudoku
cd SnakeGame/Test && python play.py        # assistir um modelo do Snake jogar
```

Para clonar só o Sudoku:

```bash
git clone --filter=blob:none --sparse https://github.com/felipe783/Teste_AI.git
cd Teste_AI
git sparse-checkout set Sudoku
```

## 🛠️ Utilitários da raiz

- `leitor_pth.py`: inspeciona um checkpoint PyTorch (camadas, shapes, nº de parâmetros e metadados).
  ```bash
  python leitor_pth.py SnakeGame/DQN/models/checkpoint.pth
  ```
- `test.py`: experimento de estudo sobre dimensões de tensores.

## 📚 Documentação

A documentação completa (arquitetura, algoritmos, decisões, problemas conhecidos e próximos passos) está no **Obsidian de contextos gerais**, no projeto **Teste_AI**. Ela também é versionada neste repositório em [`vault/`](vault/_Teste_AI.md); comece pelo índice `_Teste_AI.md`.
