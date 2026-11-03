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
Atualizado: 2026-11-03
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
| Snake DQN local com pygame e vídeo ([[TA-RF 05]]) | Removido em 2026-10-29 (pasta `SnakeGame/DQN/`, v1 de 11 estados) |
| Snake GA local headless ([[TA-RF 06]]) | Removido em 2026-10-29; o algoritmo só existe em `Server_GA`, que não executa |
| Snake DQN em servidor Docker ([[TA-RF 07]]) | Implementado (versão 2: Double DQN, 21 estados) |
| Snake GA em servidor Docker ([[TA-RF 07]]) | Quebrado — imports, `CMD` e volumes incorretos |
| Teste dos modelos com gravação de recordes ([[TA-RF 08]]) | Implementado para o DQN v2; opção do GA incompatível |
| Snake DQN v2: melhor modelo por avaliação com sementes fixas e comparação pareada | Implementado em 2026-10-17 (`best_model.pth`, `eval_log.csv`, `--eval --model A B`) |
| Testes automatizados | Só DQN v2: `SnakeGame/Server/Server_DQN/tests/test_dqn.py` (6 testes, `unittest`) |
| Snake DQN v2: auditoria técnica ([[Auditoria Snake DQN 2026-11-03]]) | Feita em 2026-11-03: algoritmo correto, platô em ~100 pontos desde ~37 000 partidas, 2 problemas críticos de checkpoint |

## Como rodar

```bash
# Ambiente (na raiz do repositório)
python -m venv .venv
.venv\Scripts\activate                  # Linux: source .venv/bin/activate
pip install -r SnakeGame/requirements.txt

# Sudoku (só biblioteca padrão)
cd Sudoku && python main.py

# Snake — os scripts usam imports "soltos": rode de dentro da pasta
cd SnakeGame/Server/Server_DQN && python DQN/agent.py           # DQN v2 sem Docker
cd SnakeGame/Server/Server_DQN && python DQN/agent.py --eval    # avaliação, 100 partidas
cd SnakeGame/Server/Server_DQN && python DQN/agent.py --eval --episodes 50 --model A.pth B.pth   # comparação pareada
cd SnakeGame/Server/Server_DQN && python -m unittest discover -s tests -v                         # testes do DQN v2
cd SnakeGame/Test && python play.py              # assistir o modelo jogar e gravar recordes

# Servidor (Linux com Docker)
bash SnakeGame/Server/Server_DQN/run_docker.sh
docker logs -f Snake_DQN
```

> `Models/`, `models/`, `logs/`, `*.json` e `*.mp4` estão no `.gitignore`: um clone novo não traz modelos treinados (os únicos versionados eram os da v1, removidos em 2026-10-29).
> Em 2026-10-17 a `.venv` local tinha `torch 2.13.0+cpu` e `pygame 2.6.1`, sem `matplotlib`. Desde 2026-10-29 nenhum script usa `matplotlib` nem `ipython` (eram do `Helper.py` dos treinos locais removidos), mas os dois continuam em `SnakeGame/requirements.txt`.

## Mapa das notas

- [[Visão Geral (Teste_AI)]]: visão resumida do repositório e dos módulos.
- [[Resumo Principal (Teste_AI)]]: o que cada subprojeto faz e os fluxos de execução.
- [[Estrutura (Teste_AI)]]: árvore de pastas e arquivos.
- [[Dependencias (Teste_AI)]]: bibliotecas e instalação.
- Requisitos: [[TA-RF 01]] · [[TA-RF 02]] · [[TA-RF 03]] · [[TA-RF 04]] · [[TA-RF 05]] · [[TA-RF 06]] · [[TA-RF 07]] · [[TA-RF 08]]
- Módulos:
  - [[!Modulo Sudoku]]: [[Main Sudoku]], [[Geração do Tabuleiro]], [[Resolvedores]], [[Tabuleiro e Validação]], [[Sudoku Samurai]], [[Logs Sudoku]].
  - [[!Modulo SnakeGame]]: [[Ambiente Snake]] (as 3 versões do jogo).
    - [[!Modulo DQN]] (v1, código removido; notas históricas): [[Agente DQN]], [[Rede e QTrainer]].
    - [[!Modulo GA]] (código só em `Server_GA`): [[Agente Genético]], [[Rede Genética]], [[Treino Genético]].
    - [[!Modulo Server]]: [[Server DQN]], [[Server GA]], [[Docker]].
    - [[!Modulo Test]]: [[Play]].
  - [[!Modulo Ferramentas]]: [[Leitor PTH]].
- Auditorias (`vault/Auditorias/`, relatórios históricos: não reescreva, marque o andamento aqui no índice): [[Auditoria Snake DQN 2026-11-03]].

## Problemas conhecidos

Encontrados lendo o código em 2026-10-17. Os itens marcados com ✅ foram **confirmados executando** uma cópia temporária; os demais não foram executados.

- ✅ Sudoku, método 1: o resultado é uma tupla `(board, bool)` e `if result:` é sempre verdadeiro. A execução mostrou "✔ RESOLVIDO" com apenas 3/9 quadrantes corretos. O método 2 retorna o próprio `board` como `result`, então também nunca vira "impossível". Ver [[Main Sudoku]].
- ✅ Sudoku Samurai: a grade tem **621** células, mas as reveladas são calculadas sobre 33² = 1089: 544, 435 e 326 células, ou seja, 88%, 70% e 52% em vez de 50%, 40% e 30%. Ver [[Sudoku Samurai]].
- ✅ `saveResults` sobrescreve `size = 9` e `desafio = False`, então o `results.json` não diferencia tamanhos nem o Samurai. Ver [[Logs Sudoku]].
- Sudoku: a cobertura usa `/ 9` fixo; com `SIZE = 16` passaria de 100%.
- Server GA não sobe: os imports usam `Teste_AI.SnakeGame.Server.GA_Server...` (pasta inexistente), o `Dockerfile` roda `DQN/agent.py`, os volumes apontam para `/Treinamento_DQN` e o `run_docker.sh` tem texto solto em `docker start ... Inicia o Container`. Ver [[Server GA]] e [[Docker]].
- `Test/play.py`, opção 2: carrega `best_genetic.pth` (15 entradas) numa rede de 21 entradas, e o arquivo nem existe em `Test/Models`. Ver [[Play]].
- `Test/play.py` num clone novo: `carregar_testes()` retorna `[]` quando `logs/testes.json` não existe, e `obter_recorde_anterior()` chama `.values()` nessa lista (`AttributeError`).
- Os caminhos relativos ao diretório atual (`videos/`, `logs/`, `results.json`, `Models/`) exigem rodar cada script de dentro da sua pasta.
- Código duplicado: três variações de `snake_gameai.py` (`Server_DQN/DQN/`, `Server_GA/GA/`, `Test/`).

Da [[Auditoria Snake DQN 2026-11-03]] (os códigos `DQN-NN` são os da auditoria; ✅ = reproduzido executando):
- ✅ **DQN-01 (crítico):** sem `Models/model.pth` (ou com `STATE_SIZE` diferente), o treino começa com `best_eval_mean = -1` e a primeira avaliação **sobrescreve o `best_model.pth`** com uma rede nova.
- ✅ **DQN-02 (crítico):** `model.pth` truncado gera `IndexError`, que o `load_checkpoint` não captura; o treino cai e o `--restart unless-stopped` o reinicia em loop. O `torch.save` não é atômico.
- ✅ **DQN-03:** a seleção usa só 20 partidas (IC95% ±13) sempre nas mesmas sementes; o `best_eval_mean` gravado (109,15) superestima o modelo (101,0 em sementes não usadas na seleção).
- ✅ **DQN-04:** a mesma semente só garante as primeiras ~6 comidas entre modelos diferentes; a comparação continua pareada, mas não "remove a sorte da comida".
- ✅ **DQN-05:** o termo `0,05 × Δespaço` da recompensa é praticamente nulo (mediana 0); quem decide é o ±0,10 da distância.
- **DQN-06:** retomar o treino perde o replay buffer (não é salvo).
- **DQN-07:** `Test/play.py` carrega o último checkpoint (`model.pth`), não o `best_model.pth`, e tem uma 4ª cópia do `get_state`.
- **DQN-08:** hiperparâmetros, semente e commit não ficam no checkpoint; os nomes dos `.pth` não batem com o conteúdo (`model30k.pth` tem 37 000 partidas).
- ✅ **Platô:** `model30k.pth` (37 000 partidas) e `model.pth` (62 200) empatam em ~101 pontos (diferença −0,6 ± 10,4).

## Decisões

- **Duas famílias de algoritmos no mesmo problema** (DQN vs. GA): o objetivo é comparar aprendizado por gradiente com busca evolutiva sobre a mesma rede `N → 256 → 3`.
- **Treino longo em servidor headless** (Docker + volumes): o ambiente sem pygame roda mais rápido e o container sobrevive a reinícios (`--restart unless-stopped`). Os modelos são trazidos depois para `Test/` para avaliação visual.
- **DQN v2 com estado mais rico** (BFS de espaço livre, rota até a cauda) e *reward shaping*: corrige o comportamento da v1 de se encurralar conforme cresce.
- **Remoção do DQN v1 e do GA local** (2026-10-29, decisão do autor): o DQN v2 é a versão mantida. As notas da v1 e do GA ficaram como histórico, e o código antigo está no Git (`dfbc504`).
- **Sudoku sem bibliotecas externas**: o foco é entender e comparar estratégias de busca (probabilidade ponderada × DFS com heurística MRV).

## Próximos passos sugeridos

Roadmap completo do DQN (17 itens, com classificação) na seção 10 da [[Auditoria Snake DQN 2026-11-03]]. Os primeiros:
- Proteger o `best_model.pth` (DQN-01) e tornar o checkpoint atômico (DQN-02) **antes** de qualquer treino novo.
- Trazer o `best_model.pth` do servidor e avaliá-lo com `--seed 777`; criar uma linha de base heurística para medir o valor do DQN.
- Experimentos contra o platô: γ = 0,99 com n-step, revisão da recompensa e features de armadilha (`espaço − comprimento`).

Demais:

1. Corrigir o veredito dos resolvedores do Sudoku (desempacotar a tupla e usar `checkVictory`).
2. Calcular a dificuldade do Samurai sobre as 621 células reais e gravar `size`/`desafio` reais no log.
3. Consertar o Server GA (imports relativos, `CMD`, volumes e `run_docker.sh`).
4. Corrigir `carregar_testes()` para retornar `{}` e alinhar a opção 2 do `play.py` com a rede do GA (15 entradas).
5. Extrair o ambiente Snake (três cópias) para um módulo compartilhado.
6. Tirar `matplotlib` e `ipython` de `SnakeGame/requirements.txt`, que nenhum script usa mais.
7. Adicionar testes mínimos: `isValid`, `checkVictory`, `action_analysis` e `train_step` com um batch sintético.
