---
Tipo: projeto
Projeto: Teste_AI
Status: ativo
Stack:
  - Python
  - PyTorch
  - pygame
  - OpenCV
  - Docker
Codigo: "Teste_AI"
Repositorio: "https://github.com/felipe783/Teste_AI"
Atualizado: 2026-10-17
---
# Teste_AI

Repositório de **experimentos com algoritmos de IA aplicados a jogos**. Reúne dois subprojetos independentes:
- **SnakeGame**: uma IA aprende a jogar Snake por **Aprendizado por Reforço (DQN)** e por **Algoritmo Genético (GA)**, com PyTorch.
- **Sudoku**: geração e resolução de Sudoku tradicional (NxN) e Samurai com **lógica pura**: probabilidade ponderada e Backtracking.

- **Código:** raiz deste repositório; a documentação fica em `vault/`
- **Entrada:** terminal (menus interativos no Sudoku e no teste do Snake; scripts de treino no SnakeGame)
- **Saída:** tabuleiros e métricas no terminal, `results.json` (Sudoku), checkpoints `.pth`, logs (`.json`/`.csv`) e vídeos `.mp4` (SnakeGame)
- **Origem:** o ambiente Snake parte do repositório [vedantgoswami/SnakeGameAI](https://github.com/vedantgoswami/SnakeGameAI)

## Status

| Funcionalidade | Estado |
|---|---|
| Sudoku tradicional NxN: geração por Backtracking randomizado ([[TA-RF 01]]) | Implementado |
| Sudoku: resolução por probabilidade (1 e N tentativas) e Backtracking ([[TA-RF 02]]) | Implementado — veredito "resolvido" incorreto nos métodos probabilísticos |
| Sudoku Samurai 33x33 ([[TA-RF 03]]) | Implementado — dificuldade calculada sobre 33², não sobre as células reais |
| Sudoku: método CSP (opção 4) | Planejado — apenas `pass` no código |
| Métricas e histórico em `results.json` ([[TA-RF 04]]) | Implementado — `size` e `desafio` sempre gravados como `9`/`False` |
| Snake DQN local com pygame e vídeo ([[TA-RF 05]]) | Implementado (versão 1, 11 estados) |
| Snake GA local headless ([[TA-RF 06]]) | Implementado (15 estados, mutação auto-adaptativa) |
| Snake DQN em servidor Docker ([[TA-RF 07]]) | Implementado (versão 2: Double DQN, 21 estados) |
| Snake GA em servidor Docker ([[TA-RF 07]]) | Quebrado — imports, `CMD` e volumes incorretos |
| Teste dos modelos com gravação de recordes ([[TA-RF 08]]) | Implementado para o DQN v2; opção do GA incompatível |
| Snake DQN v2: melhor modelo por avaliação com sementes fixas e comparação pareada | Implementado em 2026-10-17 (`best_model.pth`, `eval_log.csv`, `--eval --model A B`) |
| Testes automatizados | Só DQN v2: `SnakeGame/Server/Server_DQN/tests/test_dqn.py` (6 testes, `unittest`) |

## Como rodar

```bash
# Ambiente (na raiz do repositório)
python -m venv .venv
.venv\Scripts\activate                  # Linux: source .venv/bin/activate
pip install -r SnakeGame/requirements.txt

# Sudoku (só biblioteca padrão)
cd Sudoku && python main.py

# Snake — os scripts usam imports "soltos": rode de dentro da pasta
cd SnakeGame/DQN && python agent.py              # treino DQN v1 (abre janela pygame)
cd SnakeGame/GA && python train_genetic.py       # treino genético (headless + gráfico)
cd SnakeGame/Server/Server_DQN && python DQN/agent.py           # DQN v2 sem Docker
cd SnakeGame/Server/Server_DQN && python DQN/agent.py --eval    # avaliação, 100 partidas
cd SnakeGame/Server/Server_DQN && python DQN/agent.py --eval --episodes 50 --model A.pth B.pth   # comparação pareada
cd SnakeGame/Server/Server_DQN && python -m unittest discover -s tests -v                         # testes do DQN v2
cd SnakeGame/Test && python play.py              # assistir o modelo jogar e gravar recordes

# Servidor (Linux com Docker)
bash SnakeGame/Server/Server_DQN/run_docker.sh
docker logs -f Snake_DQN
```

> `Models/`, `models/`, `logs/`, `*.json` e `*.mp4` estão no `.gitignore`: um clone novo não traz modelos treinados, exceto `SnakeGame/DQN/models/*.pth`, que foram versionados antes da regra.
> Em 2026-10-17 a `.venv` local tinha `torch 2.13.0+cpu` e `pygame 2.6.1`, mas **não tinha `matplotlib`**, necessário para os treinos locais (Helper.py).

## Mapa das notas

- [[Visão Geral (Teste_AI)]]: visão resumida do repositório e dos módulos.
- [[Resumo Principal (Teste_AI)]]: o que cada subprojeto faz e os fluxos de execução.
- [[Estrutura (Teste_AI)]]: árvore de pastas e arquivos.
- [[Dependencias (Teste_AI)]]: bibliotecas e instalação.
- Requisitos: [[TA-RF 01]] · [[TA-RF 02]] · [[TA-RF 03]] · [[TA-RF 04]] · [[TA-RF 05]] · [[TA-RF 06]] · [[TA-RF 07]] · [[TA-RF 08]]
- Módulos:
  - [[!Modulo Sudoku]]: [[Main Sudoku]], [[Geração do Tabuleiro]], [[Resolvedores]], [[Tabuleiro e Validação]], [[Sudoku Samurai]], [[Logs Sudoku]].
  - [[!Modulo SnakeGame]]: [[Ambiente Snake]] (as 4 versões do jogo).
    - [[!Modulo DQN]]: [[Agente DQN]], [[Rede e QTrainer]].
    - [[!Modulo GA]]: [[Agente Genético]], [[Rede Genética]], [[Treino Genético]].
    - [[!Modulo Server]]: [[Server DQN]], [[Server GA]], [[Docker]].
    - [[!Modulo Test]]: [[Play]].
  - [[!Modulo Ferramentas]]: [[Leitor PTH]].

## Problemas conhecidos

Encontrados lendo o código em 2026-10-17. Os itens marcados com ✅ foram **confirmados executando** uma cópia temporária; os demais não foram executados.

- ✅ Sudoku, método 1: o resultado é uma tupla `(board, bool)` e `if result:` é sempre verdadeiro. A execução mostrou "✔ RESOLVIDO" com apenas 3/9 quadrantes corretos. O método 2 retorna o próprio `board` como `result`, então também nunca vira "impossível". Ver [[Main Sudoku]].
- ✅ Sudoku Samurai: a grade tem **621** células, mas as reveladas são calculadas sobre 33² = 1089: 544, 435 e 326 células, ou seja, 88%, 70% e 52% em vez de 50%, 40% e 30%. Ver [[Sudoku Samurai]].
- ✅ `saveResults` sobrescreve `size = 9` e `desafio = False`, então o `results.json` não diferencia tamanhos nem o Samurai. Ver [[Logs Sudoku]].
- Sudoku: a cobertura usa `/ 9` fixo; com `SIZE = 16` passaria de 100%.
- DQN v1: `get_action` recalcula `epsilon = max(5, 80 - n_game)`, ignorando o valor carregado do checkpoint. Depois de 75 partidas a exploração fica fixa em ~2,5% (5/201). Ver [[Agente DQN]].
- DQN v1 não tem *target network*: o alvo de Bellman é calculado pela mesma rede que está sendo treinada. Ver [[Rede e QTrainer]].
- DQN v1 treina com lote de 1 a cada passo do ambiente (`train_short_memory`), o mesmo problema corrigido na v2 em 2026-10-17. **A v1 não foi alterada.** Ver [[Server DQN#Regressão 15k × 30k (2026-10-17)]].
- ✅ (corrigido 2026-10-17) DQN v2: ~99,8% dos passos de gradiente eram online com lote 1; o *replay* entrava com 1 passo por partida. `model.pth` era sobrescrito sem nenhuma avaliação, então não existia "melhor modelo" guardado. Ver [[Server DQN#Regressão 15k × 30k (2026-10-17)]].
- Server GA não sobe: os imports usam `Teste_AI.SnakeGame.Server.GA_Server...` (pasta inexistente), o `Dockerfile` roda `DQN/agent.py`, os volumes apontam para `/Treinamento_DQN` e o `run_docker.sh` tem texto solto em `docker start ... Inicia o Container`. Ver [[Server GA]] e [[Docker]].
- `Test/play.py`, opção 2: carrega `best_genetic.pth` (15 entradas) numa rede de 21 entradas, e o arquivo nem existe em `Test/Models`. Ver [[Play]].
- `Test/play.py` num clone novo: `carregar_testes()` retorna `[]` quando `logs/testes.json` não existe, e `obter_recorde_anterior()` chama `.values()` nessa lista (`AttributeError`).
- Os caminhos relativos ao diretório atual (`videos/`, `logs/`, `results.json`, `Models/`) exigem rodar cada script de dentro da sua pasta.
- `DQN/snake_gameai.py` captura todos os frames de cada partida em memória para vídeo, mesmo quando o vídeo não é salvo.
- Código duplicado: `GA/` ≈ `Server_GA/GA/`, `Helper.py` idêntico em DQN e GA, e quatro variações de `snake_gameai.py`.

## Decisões

- **Duas famílias de algoritmos no mesmo problema** (DQN vs. GA): o objetivo é comparar aprendizado por gradiente com busca evolutiva sobre a mesma rede `N → 256 → 3`.
- **Treino longo em servidor headless** (Docker + volumes): o ambiente sem pygame roda mais rápido e o container sobrevive a reinícios (`--restart unless-stopped`). Os modelos são trazidos depois para `Test/` para avaliação visual.
- **DQN v2 com estado mais rico** (BFS de espaço livre, rota até a cauda) e *reward shaping*: corrige o comportamento da v1 de se encurralar conforme cresce.
- **Sudoku sem bibliotecas externas**: o foco é entender e comparar estratégias de busca (probabilidade ponderada × DFS com heurística MRV).

## Próximos passos sugeridos

1. Corrigir o veredito dos resolvedores do Sudoku (desempacotar a tupla e usar `checkVictory`).
2. Calcular a dificuldade do Samurai sobre as 621 células reais e gravar `size`/`desafio` reais no log.
3. Consertar o Server GA (imports relativos, `CMD`, volumes e `run_docker.sh`).
4. Corrigir `carregar_testes()` para retornar `{}` e alinhar a opção 2 do `play.py` com a rede do GA (15 entradas).
5. Extrair o ambiente Snake e o `Helper.py` para um módulo compartilhado.
6. Adicionar testes mínimos: `isValid`, `checkVictory`, `action_analysis` e `train_step` com um batch sintético.
