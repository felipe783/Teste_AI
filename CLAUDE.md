# CLAUDE.md — Teste_AI

Repositório de **experimentos de IA em jogos**, com dois subprojetos **independentes** (não compartilham código):
- **SnakeGame**: rede `N → 256 → 3` treinada por **DQN** (v1 local com pygame; v2 *Double DQN* para servidor Docker) e por **Algoritmo Genético**.
- **Sudoku**: geração e resolução NxN e Samurai 33x33 com **lógica pura** (probabilidade ponderada e Backtracking com MRV), só com a biblioteca padrão.

O perfil do autor e as preferências gerais estão no `CLAUDE.md` de `!Faculdade`. Este arquivo trata só do que é específico deste repositório.

## Documentação (fonte de verdade)

O índice abaixo é carregado em toda sessão. As notas detalhadas ficam em `vault/` e devem ser lidas **sob demanda**:

@vault/_Teste_AI.md

| Assunto | Nota |
|---|---|
| As 4 versões do jogo, `action_analysis`, *reward shaping* | `vault/modulos/SnakeGame/Ambiente Snake.md` |
| DQN v1 (estado de 11 entradas, ε, treino) | `vault/modulos/SnakeGame/DQN/` |
| DQN v2 (21 entradas, target network, Double DQN, CSV) | `vault/modulos/SnakeGame/Server/Server DQN.md` |
| GA (fitness, crossover, mutação auto-adaptativa) | `vault/modulos/SnakeGame/GA/` |
| Docker e Server GA quebrado | `vault/modulos/SnakeGame/Server/Docker.md`, `Server GA.md` |
| Teste visual e gravação de recordes | `vault/modulos/SnakeGame/Test/Play.md` |
| Sudoku: fluxo, geradores, resolvedores, Samurai, logs | `vault/modulos/Sudoku/` |

Conceitos transversais (avaliação de modelos, caminhos, Docker, testes…), com aprendizados deste e de outros projetos: `E:/Downloads/!Faculdade/Contexto Projetos/Conhecimento/_Conhecimento.md`. Leia sob demanda, quando a tarefa tocar o assunto.

## Mapa do código

- `SnakeGame/DQN/`: DQN v1 local. `agent.py` treina, `model.py` tem `Linear_QNet` + `QTrainer` (sem target network), `snake_gameai.py` usa pygame e grava vídeo.
- `SnakeGame/GA/`: `train_genetic.py` (população 80, elite 15%, torneio 5), `agent.py` (estado de 15 entradas, fitness `score² × 100 + passos`), `model.py` (pesos como genoma).
- `SnakeGame/Server/Server_DQN/`: **DQN v2**, a versão mais avançada; `Dockerfile` + `run_docker.sh`.
- `SnakeGame/Server/Server_GA/`: cópia do GA para Docker (**não executa**).
- `SnakeGame/Test/`: `play.py` carrega `Models/model.pth` (DQN v2, 21 entradas) e grava só as partidas recordistas.
- `Sudoku/`: `main.py` (menus e métricas), `Sudoku/src/` (regras, geração, resolvedores, logs), `Sudoku/src/SamuraiGame/`, `Sudoku/core/config.py` (posições dos 9 tabuleiros).
- Raiz: `leitor_pth.py` (inspeciona checkpoints), `test.py` (estudo de tensores).

## Comandos

Os scripts importam módulos vizinhos sem pacote (`from snake_gameai import ...`, `from src.X import *`), então **rode cada um de dentro da sua pasta**:

```powershell
.\.venv\Scripts\Activate.ps1

cd Sudoku; python main.py                                   # interativo
cd SnakeGame\Server\Server_DQN; python DQN\agent.py --eval --episodes 20   # avalia o DQN v2 sem treinar
cd SnakeGame\Test; python play.py                           # janela pygame + vídeo
python leitor_pth.py SnakeGame\DQN\models\checkpoint.pth    # da raiz
```

Para testar o Sudoku sem interação, use uma **cópia fora do repositório**, porque cada execução acrescenta um registro em `results.json`. O terminal do Windows precisa de `PYTHONIOENCODING=utf-8` quando a saída é redirecionada:

```bash
printf '1\n9\n2\n3\n3\n' | PYTHONIOENCODING=utf-8 python main.py   # tradicional 9x9, 2 jogos, difícil, backtracking
```

Os treinos locais precisam de `matplotlib` (`Helper.py`). **Peça autorização antes de instalar.** O que a `.venv` tinha na última verificação está no "Como rodar" do índice. Só o DQN v2 tem testes: `cd SnakeGame\Server\Server_DQN; python -m unittest discover -s tests -v` (~1 min; rodam em diretório temporário).

## Regras deste projeto

- **Não inicie treinos** (`DQN/agent.py`, `GA/train_genetic.py`, `Server_DQN/DQN/agent.py` sem `--eval`) sem pedir. Eles rodam indefinidamente e **sobrescrevem checkpoints**.
- **Nunca sobrescreva modelos treinados** (`DQN/models/*.pth`, `Test/Models/*.pth`, `GA/Models/*.pth`). Ao avaliar, prefira `--eval` ou o `leitor_pth.py`.
- Saídas usam **caminhos relativos ao diretório atual** (`videos/`, `logs/`, `Models/`, `results.json`). Rodar da pasta errada espalha arquivos pelo repositório.
- **Compatibilidade de checkpoint = número de entradas**: v1 tem 11, GA tem 15, v2/Test têm 21. Ao mudar `get_state`, mude `STATE_SIZE` e documente: o `load_checkpoint` do v2 recusa tamanhos diferentes e começa uma rede nova.
- **Código duplicado por cópia**: `GA/` ≈ `Server_GA/GA/`, `Helper.py` em DQN e GA, e 4 variações de `snake_gameai.py`. Ao corrigir algo, diga quais cópias ficaram de fora em vez de supor que a correção vale para todas.
- Sudoku é **só biblioteca padrão**: não introduza dependências.
- No Sudoku, os resolvedores retornam formatos diferentes (`(board, bool)`, `(board, n, bool)`, `(board, True)` ou `False`). Verifique o tipo antes de usar o resultado.
- `.gitignore` ignora `Models/`, `models/`, `logs/`, `*.json` e `*.mp4`: modelos, históricos e vídeos não vão para o Git.

## Armadilhas conhecidas

Estão em **Problemas conhecidos** do índice importado acima (`vault/_Teste_AI.md`), que é a única lista mantida.

## Manter a documentação

Ao alterar comportamento, atualize na mesma tarefa:
1. `vault/_Teste_AI.md`: **Status**, **Problemas conhecidos** (marque com ✅ o que foi confirmado executando) e `Atualizado`.
2. A nota do módulo afetado.
3. O README do subprojeto (`README.md`, `SnakeGame/README.md`, `SnakeGame/Server/README.md` ou `Sudoku/README.md`), se mudar comando ou saída.

Convenções do vault (o mesmo do Obsidian em `Contexto Projetos/Teste_AI`, via junction):
- Frontmatter: siga a tabela de **Convenções** em `E:/Downloads/!Faculdade/Contexto Projetos/Contexto Geral.md`, que é a única definição (`Tipo`, `Status` controlado, `Codigo` relativo) e tem templates em `_templates/`.
- Nomes de nota únicos no vault inteiro: notas genéricas levam `(Teste_AI)`; requisitos usam `TA-RF NN`.
- Links `[[Nome da Nota]]`. Separe o que foi **executado** do que foi **lido no código**, com data.
