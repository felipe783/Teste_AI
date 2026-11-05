# 🐳 SnakeGame — Treino em Servidor

Versões do treino **sem interface gráfica**, empacotadas em Docker para rodar continuamente num servidor Linux. Os modelos, o replay buffer, os logs e a configuração ficam num volume do host, então o treino continua de onde parou mesmo que o container seja recriado.

| Pasta | Algoritmo | Estado |
|---|---|---|
| `Server_DQN/` | **DQN v2**: Double DQN, *target network* (atualizada a cada 1 000 passos), Huber loss, gradient clipping, ε linear de 1.0 a 0.02, estado de 21 entradas com análise de espaço livre por BFS; hiperparâmetros na CLI, checkpoint atômico e retomada com replay | Funcional |
| `Server_GA/` | Algoritmo Genético (única cópia; a versão local `../GA` foi removida em 2026-10-29) | **Não executa ainda**: imports, `CMD` do Dockerfile e volumes precisam de ajuste |

## 🚀 Como rodar

### Com Docker (servidor Linux)

```bash
bash SnakeGame/Server/Server_DQN/run_docker.sh                         # treino sem fim em BASE_DIR
CONTAINER_NAME=gamma099 BASE_DIR=~/runs/gamma099 RESTART_POLICY=on-failure:5 \
    bash SnakeGame/Server/Server_DQN/run_docker.sh --gamma 0.99 --seed 1 --games 20000   # experimento
```

O script:
1. constrói a imagem `snake_dqn` (torch só de CPU; `SKIP_BUILD=1` pula);
2. para (`docker stop -t 120`, que salva antes de sair) e remove o container antigo;
3. cria o container com `--restart $RESTART_POLICY` (padrão `unless-stopped`), `--cpus 1`, o usuário do host e o volume `BASE_DIR → /data` (padrão `/home/felipe/Teste_IA/Server_DQN`), com `Models/`, `logs/` e `config.json`;
4. repassa os argumentos extras ao `DQN/agent.py`.

> Volumes criados pela versão antiga (container como root): rode uma vez `sudo chown -R $(id -u):$(id -g) "$BASE_DIR"`; o script avisa.

Acompanhar:

```bash
docker logs -f Snake_DQN | grep -v '^Partida'      # linhas [status] e avaliações
column -s, -t < "$BASE_DIR/logs/eval_log.csv" | tail
```

### Na AWS (EC2)

Os scripts de `Server_DQN/aws/` preparam a instância (`setup_ec2.sh`), medem a vazão (`benchmark.sh`), lançam um lote de experimentos em paralelo com desligamento automático opcional (`experiments.sh`) e resumem o andamento (`status.sh`). A instância recomendada, os custos e o plano para os créditos estão em [`vault/modulos/SnakeGame/Server/Treino na AWS.md`](<../../vault/modulos/SnakeGame/Server/Treino na AWS.md>).

### Sem Docker

```bash
cd SnakeGame/Server/Server_DQN
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 && pip install -r requirements.txt
python DQN/agent.py                                    # treina (ou retoma) em ./Models e ./logs
python DQN/agent.py --run-dir runs/x --seed 0 --games 15000   # execução isolada e reprodutível
python DQN/agent.py --help                             # hiperparâmetros (--gamma, --n-step, --safe-explore...)
python DQN/agent.py --eval --episodes 100              # avalia sem explorar nem salvar
python DQN/agent.py --eval --episodes 50 --seed 777 --model Models/a.pth Models/b.pth   # comparação pareada
python -m unittest discover -s tests -v                # 21 testes, ~10 s
```

O treino grava, dentro do `--run-dir`:
- `Models/model.pth`: último estado, para retomar, a cada 100 partidas ou 10 min;
- `Models/best_model.pth`: melhor média em 50 partidas gulosas com sementes fixas, avaliadas a cada 500 partidas; só é substituído quando a média melhora;
- `Models/replay.npz`: o replay buffer;
- `logs/training_log.csv` e `logs/eval_log.csv` (com IC95%);
- `config.json`: os hiperparâmetros da execução, reaproveitados ao retomar.

Toda gravação é atômica, com cópia `.bak`. `SIGTERM`/`Ctrl+C` param salvando tudo. Um checkpoint ilegível sem `.bak`, ou de outro número de entradas, faz o treino **parar com código 2** sem sobrescrever nada.

> Checkpoints anteriores a 2026-10-17 carregam normalmente, mas o `train_steps` deles contava 1 por passo do ambiente; agora conta 1 por lote (a cada 4 passos). Em checkpoints longos, o ε já está no mínimo, então não muda nada.

### Levar o modelo para assistir

Copie `Models/best_model.pth` (ou `model.pth`) do servidor para `SnakeGame/Test/Models/` e rode `python play.py` dentro de `SnakeGame/Test`; a opção 1 usa o `best_model.pth` quando ele existe.

## 📚 Documentação

A documentação completa do projeto está em [`vault/_Teste_AI.md`](../../vault/_Teste_AI.md). Comece por esse arquivo para consultar a arquitetura, os algoritmos, as decisões técnicas, os problemas conhecidos e os próximos passos. Os detalhes do servidor estão em [`vault/modulos/SnakeGame/Server/`](<../../vault/modulos/SnakeGame/Server/!Modulo Server.md>). A auditoria técnica do DQN v2 (problemas conhecidos, limitações e roadmap) está em [`vault/Auditorias/Auditoria Snake DQN 2026-11-03.md`](<../../vault/Auditorias/Auditoria Snake DQN 2026-11-03.md>).
