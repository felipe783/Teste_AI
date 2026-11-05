# Correções do Snake DQN e preparação para a AWS — 2026-11-05

Continuação da [[Auditoria Snake DQN 2026-11-03]]: cada apontamento foi revalidado no código, os problemas de engenharia foram corrigidos e o treino foi preparado para rodar numa EC2. A infraestrutura recomendada e o plano para os US$ 100 de créditos estão em [[Treino na AWS]].

Escopo: `SnakeGame/Server/Server_DQN/` ([[Server DQN]], [[Docker]], [[Ambiente Snake]]) e duas correções pequenas em `SnakeGame/Test/play.py` ([[Play]]). Este relatório é histórico; o andamento fica no índice [[_Teste_AI]].

Marcadores: **[CONFIRMADO 2026-11-05]** executado ou medido · **[LIDO 2026-11-05]** conclusão da leitura do código · **[PROVÁVEL]** dedução não testada.

## 1. Resumo da auditoria original

A auditoria de 2026-11-03 concluiu que o **núcleo do DQN está correto** (Double DQN, target network, Huber, clipping, máscara `~done`) e que o agente **estagnou em ~100 pontos** desde ~37 000 partidas. Ela apontou 2 problemas críticos de checkpoint (DQN-01, DQN-02), uma avaliação ruidosa (DQN-03), problemas de rastreabilidade (DQN-08), de eficiência (95% do tempo em BFS) e de higiene (DQN-09), além de 4 limitações algorítmicas (L1–L4: horizonte curto, recompensa, representação, falta de linha de base).

## 2. Validação dos apontamentos no código atual

O código do DQN não tinha mudado desde a auditoria (o último commit, `43da278`, só tocou na documentação). Mesmo assim, os itens foram reproduzidos em vez de presumidos, com a `.venv` local: torch 2.13.0, numpy 2.5.2 e Python 3.12.

| Item | Situação em 2026-11-05, antes da correção |
|---|---|
| DQN-01 `best_model` sobrescrito | **[CONFIRMADO]** Num diretório temporário, sem `model.pth` e com o modelo de 62 200 partidas como `best_model.pth`, 2 partidas de treino deixaram o `best_model.pth` com `n_game = 2` e `best_eval_mean = 0`. |
| DQN-02 checkpoint truncado | **[CONFIRMADO, sintoma diferente]** Com o torch 2.13, truncar o `model.pth` em 1–99,9% **não derruba** o treino: o erro (inclusive `OSError: Invalid argument`) é capturado, o treino **recomeça do zero em silêncio** e sobrescreve o arquivo no próximo salvamento. É pior que o crash relatado, porque não deixa sinal. O `torch.save` continuava não atômico. |
| DQN-03 avaliação ruidosa | **[CONFIRMADO]** O `best_model` antigo gravou 109,15 com 20 partidas; reavaliado no protocolo novo, deu 95,3 (10 partidas) e o modelo fez 101,0 ± 10,3 em 40 partidas nas sementes 777. É a maldição do vencedor. |
| DQN-04 sementes | **[LIDO]** Inalterado: as comidas só coincidem até os corpos divergirem. |
| DQN-05 termo de espaço nulo | **[LIDO]** Inalterado (a recompensa não foi mudada; ver §7). |
| DQN-06 replay perdido | **[CONFIRMADO]** O buffer não era salvo. O efeito de retomar com buffer vazio foi **inconclusivo**: em 2 sementes, 25 partidas de treino e 20 avaliações, a média caiu 12 pontos numa semente e subiu 3 na outra. |
| DQN-07 `play.py` | **[LIDO]** Carregava o `model.pth`; a 4ª cópia do `get_state` continua lá. |
| DQN-08 rastreabilidade | **[LIDO]** Hiperparâmetros em constantes de módulo e semente não registrada. Em 2026-11-05 o `Test/Models/` tinha **só** o `model.pth` (62 200 partidas); os outros 7 `.pth` citados na auditoria não estão mais nesta máquina. |
| DQN-09 menores | **[LIDO]** Todos presentes: vitória com `food = None`, 3 BFS no estado terminal, `old_space` recalculado, `torch` com CUDA, `set_num_threads(4)` fixo, log apagado aos 5 GB. |
| Eficiência | **[CONFIRMADO]** `cProfile`: ~95% do tempo em `_reachable_area`, 5 BFS por passo; 528 passos/s no laço de treino. |

**Achados novos (não estavam na auditoria):**
- **[CONFIRMADO 2026-11-05] A exploração mata o agente na fase longa.** Num teste com o modelo de 62 200 partidas sem treinar, em 40 partidas com as sementes 777+i: **101,0 ± 10,3** guloso, **30,8 ± 5,8** com ε = 0,02 e **79,4 ± 9,9** com ε = 0,02 sorteando só entre ações que não colidem. Cerca de 70% das mortes nas partidas de treino vêm de ações aleatórias suicidas. Por isso o treino quase nunca vê a fase longa, que é onde o agente falha, e o `training_log.csv` subestima o agente em ~3×. [PROVÁVEL] É uma causa do platô complementar ao L1. Essa hipótese motiva o lote 1 de [[Treino na AWS]].
- **[CONFIRMADO 2026-11-05] 1 thread é tão rápida quanto 4:** 535 contra 509 passos/s. A rede tem ~6 mil parâmetros, então escalar é rodar **processos em paralelo**.
- **[LIDO 2026-11-05]** O `new_space` do `play_step` é exatamente o BFS que o `action_analysis` já tinha feito para a ação escolhida, porque a cabeça nova e o corpo sem as duas últimas células são os mesmos. Eram 2 BFS redundantes por passo, não 1.
- **[LIDO 2026-11-05]** O container rodava como root e o `config` não existia; com `docker stop` (SIGTERM), até 100 partidas de progresso se perdiam, porque só o `Ctrl+C` salvava.

## 3. Problemas corrigidos

| Item | Correção | Evidência |
|---|---|---|
| DQN-01 | O `best_eval_mean` vem do **próprio** `best_model.pth` (`protect_best`), nunca do `model.pth`. Um `best_model` ilegível ou de outro `STATE_SIZE` é arquivado (renomeado), não sobrescrito. | Teste `test_missing_model_does_not_overwrite_best` |
| DQN-02 | `atomic_write`: grava em `.tmp`, faz `fsync` e `os.replace`, e a versão anterior vira `.bak`. Ao retomar, um arquivo ilegível cai no `.bak` (o ilegível é preservado como `model.corrompido-<data>.pth`). Sem `.bak` válido, ou com `STATE_SIZE` diferente, o treino **para com código 2 e não toca em nada**. | Testes `test_truncated_checkpoint_falls_back_to_bak` e `test_corrupt_without_bak_or_incompatible_stops_without_touching`; CLI: `exit=2` com arquivo intacto |
| DQN-03 | A avaliação de seleção passou de 20 para **50 partidas** (`--eval-episodes`), com IC95% no `eval_log.csv`. O protocolo (`episodes`, `seed`) vai no checkpoint, e um `best_model` medido com outro protocolo é **reavaliado** ao iniciar. | `test_best_from_other_protocol_is_reevaluated`; ponta a ponta: 109,15 → 95,3 |
| DQN-04 | Texto corrigido no `run_eval` e na comparação pareada ("reduz a variância", e não "remove a sorte"). | — |
| DQN-06 | O replay buffer é salvo (`Models/replay.npz`, ~18 MB com 100 mil transições) nos checkpoints periódicos e na parada. Sem ele, o treino coleta 20 000 transições (`--resume-warmup`) antes do primeiro gradiente. | `test_resume_continues_counters_replay_and_config`; ponta a ponta: "Replay buffer restaurado: 100000 transições" |
| DQN-07 (parcial) | `play.py`, opção 1: usa o `best_model.pth` quando existe, senão o `model.pth`. | Lido; o `play.py` não tem teste |
| DQN-08 | `Config` (dataclass) com todos os hiperparâmetros na CLI. Cada execução tem `--run-dir` com `config.json`, `Models/` e `logs/`. A semente é sempre registrada (sorteada se omitida), e o checkpoint leva `config`, `git_commit`, `env_steps`, `train_seconds` e `saved_at`. | `config.json` ponta a ponta; commit `43da278-dirty` no checkpoint |
| DQN-09 | Sem `get_state` no estado terminal, o que também elimina o crash da vitória. O `training_log.csv` é **rotacionado** aos 256 MB, nunca apagado. Docker com torch só de CPU, `--threads` (padrão 1) e o `set_num_threads(4)` fixo removido. | `test_filling_the_board_ends_with_victory_not_crash` |
| Eficiência | BFS em grade de inteiros com `bytearray`, cache das 3 análises por estado e reaproveitamento do espaço: 5 → 3 BFS por passo. Replay em arrays NumPy pré-alocados (sorteio O(lote)). | Ver §6 |
| Parada limpa | `SIGTERM` e `Ctrl+C` param no próximo passo e salvam modelo e replay; uma avaliação em andamento é interrompida. Há checkpoint também a cada 10 min (`--checkpoint-minutes`). | `test_sigterm_stops_cleanly_and_saves`; `docker stop` salvou em 0,7 s |
| `play.py` | `carregar_testes()` devolvia `[]` sem `logs/testes.json` (`AttributeError` num clone novo); agora devolve `{}`. | Lido |

## 4. Arquivos modificados

| Arquivo | Mudança e justificativa |
|---|---|
| `Server_DQN/DQN/agent.py` | `Config`, `RunPaths`, `ReplayBuffer`, `NStepAccumulator`, `atomic_write`, `read_checkpoint`, `resume`, `protect_best`, `StopRequest` e uma CLI com hiperparâmetros. O `get_state`, o ε, o `get_action` (sem `--safe-explore`) e os comentários didáticos foram mantidos. |
| `Server_DQN/DQN/snake_gameai.py` | BFS em inteiros com cache, sem mudar nenhum valor (estado, recompensa, comida e RNG idênticos; ver §6). O comentário de "diferenças de potencial" foi corrigido (L2). |
| `Server_DQN/DQN/model.py` | O `train_step` aceita índices de ação além do one-hot; saiu o `set_num_threads(4)` executado na importação. |
| `Server_DQN/tests/test_dqn.py` | De 6 para 21 testes; os 6 antigos foram adaptados à API nova sem mudar o que verificam. |
| `Server_DQN/Dockerfile`, `.dockerignore`, `requirements.txt` | torch 2.13.0 de CPU (amd64 e arm64), versões fixas, `ENTRYPOINT ... --run-dir /data`, `GIT_COMMIT`, 1 thread. O `.dockerignore` tira `Models/`, `logs/` e `*.pth` da imagem. |
| `Server_DQN/run_docker.sh` | Variáveis `BASE_DIR`, `CONTAINER_NAME`, `RESTART_POLICY`, `CPUS` e `SKIP_BUILD`; argumentos repassados ao `agent.py`; um volume único em `/data` (o `config.json` também persiste); `--stop-timeout 120`; `--user` do host; logs do Docker com 50 MB × 3. |
| `Server_DQN/aws/` (novo) | `setup_ec2.sh`, `benchmark.sh`, `experiments.sh` (lote paralelo, com `AUTO_SHUTDOWN=1` opcional) e `status.sh`. |
| `Test/play.py` | `best_model.pth` primeiro; `carregar_testes()` devolve `{}`. |

**Mudanças de comportamento intencionais:**
- Um `model.pth` incompatível ou corrompido sem `.bak` agora **para** o treino. Antes, o treino recomeçava do zero e sobrescrevia o arquivo.
- O padrão de `eval_episodes` passou de 20 para 50.
- O `--eval` continua avaliando o `model.pth` por padrão.

**Não mudou:** os hiperparâmetros de aprendizado, a recompensa, a rede e o `get_state`. Um modelo treinado com o código novo, nas mesmas condições, aprende a mesma coisa; só a amostragem do replay usa outro gerador aleatório (NumPy).

## 5. Melhorias implementadas para os experimentos

- `--n-step N`: retornos de N passos com bootstrap `γ^N` (L1). O padrão 1 é idêntico ao DQN anterior (teste `test_one_step_is_identity`).
- `--safe-explore`: na exploração, sorteia só entre ações sem colisão imediata (achado novo da §2). Desligado por padrão.
- `--gamma`, `--lr`, `--batch-size`, `--memory`, `--train-every`, `--target-update`, `--epsilon-min` e `--epsilon-decay-steps` sem editar código; numa retomada, o valor que não for informado vem do `config.json`.
- `eval_log.csv` com `ic95`, `epsilon`, `train_steps`, `passos_ambiente`, `horas_treino` e `passos_por_s`. Um CSV com o cabeçalho antigo é arquivado, não misturado.
- Linha `[status]` no log a cada checkpoint: média das últimas 100 partidas, ε, passos/s, horas e tamanho do replay.

## 6. Resultados dos testes

Todos executados em 2026-11-05 nesta máquina (Ryzen AI 5 340, 12 threads, 14 GB).

| Teste | Resultado |
|---|---|
| Suíte original, antes das mudanças | 6/6 OK em 6,7 s |
| Suíte nova (`python -m unittest discover -s tests -v`) | **21/21 OK em ~10 s**, também com `-W error::ResourceWarning` |
| Suíte dentro do container (torch 2.13.0+cpu) | 21/21 OK em 10,9 s |
| Equivalência do ambiente: original × novo, 60 partidas com política aleatória ponderada | 17 304 passos com estado, recompensa, fim, comida e RNG idênticos; análise **5,7×** mais rápida |
| Não regressão: `run_eval` do código antigo × novo com o modelo de 62 200 partidas, sementes 777, 20 partidas | **Placares idênticos partida a partida** (média 93,50); 75,2 s → **12,6 s** |
| Laço de treino (1 thread, modelo de 62 200 partidas) | 528 → **2 395 passos/s (4,5×)** |
| Divisão do tempo depois da otimização | gradiente 37% · inferência 4% · ambiente e estado 59% |
| 4 treinos em paralelo | ~1 850 passos/s cada (~7 400 no total) |
| Memória e disco por treino (container) | ~275 MB de RAM, 1 núcleo a ~99%, ~35 MB por execução |
| Ponta a ponta pela CLI: retomar o modelo real, `SIGTERM` aos 90 s, retomar de novo | Continuou da partida 62 401, replay de 100 mil restaurado, `config.json` respeitado, `best_model` reavaliado (109,15 → 95,3) |
| Ponta a ponta no Docker (`run_docker.sh`): `--games` e `on-failure`, `docker stop` e retomada | Saiu com 0 ao atingir `--games` e não reiniciou; `docker stop` salvou em 0,7 s; arquivos do usuário do host |
| `aws/benchmark.sh` (N = 2, 45 s), `aws/experiments.sh` (`GAMES=30`) e `aws/status.sh` | Funcionaram; corrigido um bug de formatação do `status.sh` |

**Comparação de métricas:** não houve treino longo novo, porque as mudanças não alteram o algoritmo e um A/B do zero leva horas: esse é justamente o papel da AWS. O que se comparou foi a **política**, que deu placares idênticos com o mesmo modelo, e a **vazão**. Nas avaliações curtas da retomada (10 partidas cada), a média ficou entre 90 e 107, dentro do ruído do modelo de ~101.

## 7. Problemas que permanecem

- **Platô (L1–L4):** nada mudou por padrão. Os experimentos ficaram disponíveis (`--gamma`, `--n-step`, `--safe-explore`), mas **não foram executados em escala**; a melhora não está provada.
- **DQN-05 e L2 (recompensa):** inalterados. Mudar a recompensa exige A/B; os pesos não estão na CLI.
- **L3:** features de armadilha e referencial egocêntrico não implementados (mudariam o `STATE_SIZE`, agora protegido pelo DQN-01).
- **L4:** sem linha de base heurística.
- **DQN-07:** o `play.py` ainda tem a 4ª cópia do `get_state`, e o `Test/snake_gameai.py` não recebeu a otimização. Continuam 3 variações de `snake_gameai.py` (`Server_DQN`, `Server_GA`, `Test`), e só a do `Server_DQN` mudou.
- **Timeout:** `frame_iteration` cumulativo e truncamento tratado como término (sem efeito medido: 0 de 180 mortes).
- **Avaliação de seleção:** 50 partidas ainda dão IC de ~±9 e as mesmas sementes. O resultado final deve ser medido com `--eval --seed 777`.
- **RNG não salvo:** ao retomar, as sementes passam a `seed + n_game`, então uma retomada não reproduz exatamente uma execução contínua.
- **`env_steps` de checkpoints antigos começa em 0** (o campo não existia).
- **`--restart unless-stopped` com checkpoint inválido** reinicia o container com *backoff*, mas sem perder dados. Com `--games`, use `RESTART_POLICY=on-failure:5`.
- **Graviton (arm64) não foi testado:** o índice CPU da PyTorch tem as rodas, mas a imagem só foi construída em x86_64. O `aws/benchmark.sh` é o primeiro passo na instância.
- **Server GA** continua quebrado (fora do escopo).

## 8. Arquitetura atualizada do treino

```
agent.py main()  ── Config: padrões ← <run-dir>/config.json ← argumentos da CLI
   └ train(config, RunPaths(run_dir))
       ├ resume(): model.pth → .bak → do zero | CheckpointError (código 2, nada sobrescrito)
       │   └ replay.npz → ReplayBuffer, ou aquecimento de 20 000 transições
       ├ protect_best(): best_eval_mean do best_model.pth; reavalia se o protocolo mudou
       └ laço (para com SIGTERM/Ctrl+C ou --games):
           get_state (3 BFS em cache) → get_action (ε, opcionalmente safe) → play_step (0 BFS extra)
           → NStepAccumulator → ReplayBuffer (NumPy circular)
           → a cada train_every passos: QTrainer.train_step (Double DQN, γ^n, Huber, clip 10)
           → fim da partida: training_log.csv; a cada eval_every: run_eval(50, seed 12345)
             → best_model.pth (atômico) + eval_log.csv
           → a cada 100 partidas ou 10 min: model.pth + replay.npz (atômicos, com .bak) + [status]
```

Layout de uma execução: `<run-dir>/config.json`, `<run-dir>/Models/{model.pth, best_model.pth, replay.npz, *.bak}` e `<run-dir>/logs/{training_log.csv, eval_log.csv}`. Com `--run-dir .`, o padrão, é o layout antigo.

## 9. Como rodar localmente

```bash
cd SnakeGame/Server/Server_DQN
python -m unittest discover -s tests -v                                   # 21 testes, ~10 s
python DQN/agent.py --run-dir runs/teste --seed 0 --games 2000            # treino curto isolado
python DQN/agent.py --run-dir runs/teste --games 4000                     # retoma com a mesma config
python DQN/agent.py --eval --episodes 50 --seed 777 --model runs/teste/Models/best_model.pth
python DQN/agent.py --help                                                # todos os hiperparâmetros
# Docker: CONTAINER_NAME=teste BASE_DIR=$PWD/runs/docker RESTART_POLICY=on-failure:5 bash run_docker.sh --games 2000
```

`runs/` está no `.gitignore`. Para AWS, veja [[Treino na AWS]].
