FROM python:3.12-slim
WORKDIR /Treinamento_DQN
# Logs aparecem na hora no `docker logs`; 1 thread por processo (a rede é pequena: mais threads não aceleram)
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY requirements.txt .
# torch só com CPU (o índice da PyTorch tem amd64 e arm64/Graviton): a imagem com CUDA tem vários GB a mais
# e o treino não usa GPU. Copiar o requirements.txt antes do código aproveita o cache de camadas.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
# Commit do código gravado nos checkpoints (o .git não entra na imagem)
ARG GIT_COMMIT=desconhecido
ENV GIT_COMMIT=$GIT_COMMIT
# Os dados ficam no volume /data (Models/, logs/ e config.json); argumentos extras vão para o agent.py
ENTRYPOINT ["python", "DQN/agent.py", "--run-dir", "/data"]
CMD []
