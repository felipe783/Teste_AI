#!/bin/bash

set -e

IMAGE_NAME="snake_dqn"
CONTAINER_NAME="Snake_DQN"
MODELS_VOLUME="snake_models_data"
LOGS_VOLUME="snake_logs_data"

cd "$(dirname "$0")"

echo "Construindo imagem..."
docker build -t "$IMAGE_NAME" .

echo "Verificando volumes de dados (Models e logs)..."
docker volume create "$MODELS_VOLUME" >/dev/null
docker volume create "$LOGS_VOLUME" >/dev/null

echo "Verificando container antigo..."

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removendo container antigo..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME"
fi

echo "Criando container..."
docker create \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -v "$MODELS_VOLUME:/app/Models" \
    -v "$LOGS_VOLUME:/app/logs" \
    "$IMAGE_NAME"

echo "Iniciando treinamento..."
docker start "$CONTAINER_NAME"

echo ""
echo "=============================================="
echo "       SNAKE DQN INICIADO"
echo "=============================================="
echo "Container: $CONTAINER_NAME"
echo "Imagem:    $IMAGE_NAME"
echo "Volumes:   $MODELS_VOLUME -> /app/Models"
echo "           $LOGS_VOLUME -> /app/logs"
echo ""
echo "Ver logs do container:"
echo "docker logs -f $CONTAINER_NAME"
echo ""
echo "Ver conteúdo dos volumes (sem precisar do container rodando):"
echo "docker run --rm -v $MODELS_VOLUME:/data alpine ls -la /data"
echo "docker run --rm -v $LOGS_VOLUME:/data alpine ls -la /data"
echo ""