#!/bin/bash

set -e

IMAGE_NAME="snake_dqn"
CONTAINER_NAME="Snake_DQN"

cd "$(dirname "$0")"

echo "Construindo imagem..."
docker build -t "$IMAGE_NAME" .

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
    "$IMAGE_NAME"

echo "Iniciando treinamento..."
docker start "$CONTAINER_NAME"

echo ""
echo "=============================================="
echo "       SNAKE DQN INICIADO"
echo "=============================================="
echo "Container: $CONTAINER_NAME"
echo "Imagem:    $IMAGE_NAME"
echo ""
echo "Ver logs:"
echo "docker logs -f $CONTAINER_NAME"
echo ""
