# Sobre os Arquivos

`Dockerfile` e `run_docker.sh` são os **Responsáveis pela Implantação** do treino no servidor.
# Dockerfile

````dockerfile
FROM python:3.12-slim
WORKDIR /Treinamento_DQN
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "DQN/agent.py"]
````

- Copiar o `requirements.txt` **antes** do código aproveita o cache de camadas: mudar só o `.py` não reinstala o PyTorch.
- `requirements.txt`: `torch` e `numpy`. Sem índice CPU, o `pip install torch` no Linux baixa a versão com CUDA, o que deixa a imagem bem maior.

# run_docker.sh

1. `cd "$(dirname "$0")"`: o script funciona de qualquer diretório.
2. `docker build -t snake_dqn .`
3. Cria `Models/` e `logs/` em `/home/felipe/Teste_IA/Server_DQN/` no host.
4. Para e remove o container antigo, se existir.
5. `docker create` com `--restart unless-stopped`, logs limitados (`--log-opt max-size=5g`) e volumes:

````bash
-v "$MODELS_DIR:/Treinamento_DQN/Models"
-v "$LOGS_DIR:/Treinamento_DQN/logs"
````

6. `docker start` e instruções para acompanhar com `docker logs -f Snake_DQN`.

No DQN, `MODEL_DIR = "Models"` e `LOG_DIR = "logs"` são relativos ao `WORKDIR`, então caem exatamente nos volumes, e o treino continua de onde parou após recriar o container.

> [!warning] Versão GA
> - `CMD ["python", "DQN/agent.py"]` deveria ser `GA/train_genetic.py`.
> - Volumes em `/Treinamento_DQN/...` com `WORKDIR /Treinamento_GA`.
> - A linha `docker start "$CONTAINER_NAME"  Inicia o Container` passa `Inicia`, `o` e `Container` como nomes de containers. Com `set -e`, o script termina com erro.
> Ver [[Server GA]].
