#!/bin/bash

set -e

IMAGE_NAME="snake_GA"
CONTAINER_NAME="Snake_GA"
BASE_DIR="/home/felipe/Teste_IA/Server_GA"
MODELS_DIR="$BASE_DIR/Models"
LOGS_DIR="$BASE_DIR/logs"

# Caminho para executar o Script
# Nao dependen de onde vc roda o .sh
cd "$(dirname "$0")"

echo "Construindo Imagem"
docker build -t "$IMAGE_NAME" . # Encontrar o Dockerfile, vai ler o da pasta atual

echo "Verificando pastas de dados no host (Models e logs)..."
# Criar as pastar necessarias
# o -p nao quebra se ela existirem
mkdir -p "$MODELS_DIR" 
mkdir -p "$LOGS_DIR"

# Deletar container antigo se existir
echo "Verificando container antigo..."
# Vai pegar o nome de todos os Containers e ver se exite um com o nome "Snake_GA"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removendo container antigo..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true # Para o container antigo
    docker rm "$CONTAINER_NAME" # Remove o container
fi

echo "Criando container..."
docker create \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --log-opt max-size=5g \
    --log-opt max-file=1 \
    -v "$MODELS_DIR:/Treinamento_DQN/Models" \
    -v "$LOGS_DIR:/Treinamento_DQN/logs" \
    "$IMAGE_NAME"

echo "Iniciando treinamento..."
docker start "$CONTAINER_NAME"  Inicia o Container

echo ""
echo "=============================================="
echo "       SNAKE GA INICIADO"
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
