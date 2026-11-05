# Sobre os Arquivos

`Dockerfile` e `run_docker.sh` são os **Responsáveis pela Implantação** do treino no servidor.
# Dockerfile (desde 2026-11-05)

````dockerfile
FROM python:3.12-slim
WORKDIR /Treinamento_DQN
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
ARG GIT_COMMIT=desconhecido
ENV GIT_COMMIT=$GIT_COMMIT
ENTRYPOINT ["python", "DQN/agent.py", "--run-dir", "/data"]
CMD []
````

- Copiar o `requirements.txt` **antes** do código aproveita o cache de camadas: mudar só o `.py` não reinstala o PyTorch.
- **torch só de CPU**, do índice da PyTorch, que tem amd64 e arm64 (Graviton). Antes, o `pip install torch` baixava a versão com CUDA, que o treino não usa. Imagem de 1,42 GB em 2026-11-05.
- `requirements.txt` com versões fixas (`torch==2.13.0`, `numpy==2.5.2`).
- `.dockerignore`: `Models/`, `logs/`, `runs/`, `*.pth`, `*.npz`, `*.csv`. Antes, o `COPY . .` levava os `.pth` antigos para a imagem.
- Argumentos depois do nome da imagem vão para o `agent.py` (`docker run snake_dqn --gamma 0.99 ...`).

# run_docker.sh

Configurável por variáveis, com os argumentos repassados ao `agent.py`:

````bash
bash run_docker.sh                                     # padrão: Snake_DQN, BASE_DIR do servidor caseiro
CONTAINER_NAME=gamma099 BASE_DIR=~/runs/gamma099 RESTART_POLICY=on-failure:5 bash run_docker.sh --gamma 0.99 --games 20000
````

1. `docker build` com `GIT_COMMIT` (gravado nos checkpoints); `SKIP_BUILD=1` pula.
2. Cria `BASE_DIR/Models` e `BASE_DIR/logs` e **para com instrução de `chown`** se houver arquivo de outro dono (volumes criados quando o container rodava como root).
3. Para (`docker stop -t 120`, que dá tempo de salvar) e remove o container antigo.
4. `docker create` com `--restart $RESTART_POLICY` (padrão `unless-stopped`), `--stop-timeout 120`, `--cpus $CPUS` (padrão 1), `--user` do host, logs de 50 MB × 3 e **um volume só**: `BASE_DIR → /data`. Os arquivos continuam em `BASE_DIR/Models` e `BASE_DIR/logs`, e o `config.json` passa a persistir junto.

O treino continua de onde parou ao recriar o container: pesos, otimizador, contadores, ε, configuração e **replay buffer** (desde 2026-11-05). O `docker stop` envia SIGTERM, e o treino salva antes de sair (0,7 s medido). Um checkpoint ilegível sem `.bak` faz o treino sair com código 2 sem apagar nada; com `unless-stopped`, o Docker tenta de novo com espera crescente. Com `--games`, use `on-failure:5` para não reiniciar o que já terminou. Ver [[Correções Snake DQN 2026-11-05]] e, para a EC2, [[Treino na AWS]].

> [!warning] Versão GA
> - `CMD ["python", "DQN/agent.py"]` deveria ser `GA/train_genetic.py`.
> - Volumes em `/Treinamento_DQN/...` com `WORKDIR /Treinamento_GA`.
> - A linha `docker start "$CONTAINER_NAME"  Inicia o Container` passa `Inicia`, `o` e `Container` como nomes de containers. Com `set -e`, o script termina com erro.
> Ver [[Server GA]].
