#!/bin/bash

set -e

IMAGE_NAME="snake_dqn"
CONTAINER_NAME="Snake_DQN"

BASE_DIR="/home/felipe/Teste_IA/Server_DQN"
MODELS_DIR="$BASE_DIR/Models"
LOGS_DIR="$BASE_DIR/logs"

cd "$(dirname "$0")"

echo "Construindo imagem..."
docker build -t "$IMAGE_NAME" .

echo "Verificando pastas de dados no host (Models e logs)..."
mkdir -p "$MODELS_DIR"
mkdir -p "$LOGS_DIR"

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
    -v "$MODELS_DIR:/Treinamento_DQN/Models" \
    -v "$LOGS_DIR:/Treinamento_DQN/logs" \
    "$IMAGE_NAME"

echo "Iniciando treinamento..."
docker start "$CONTAINER_NAME"

echo ""
echo "=============================================="
echo "       SNAKE DQN INICIADO"
echo "=============================================="
echo "Container: $CONTAINER_NAME"
echo "Imagem:    $IMAGE_NAME"
echo "Volumes:   $MODELS_DIR -> /Treinamento_DQN/Models"
echo "           $LOGS_DIR -> /Treinamento_DQN/logs"
echo ""
echo "Ver logs do container:"
echo "docker logs -f $CONTAINER_NAME"
echo ""
echo "Ver conteúdo direto no host (não precisa do container):"
echo "ls -la $MODELS_DIR"
echo "ls -la $LOGS_DIR"
echo ""