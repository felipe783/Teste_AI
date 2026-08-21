#!/bin/bash

set -e

# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR="Teste_IA"
TRAIN_DIR="$BASE_DIR/Treinamento_DQN"

IMAGE_NAME="snake_dqn"
CONTAINER_NAME="Snake_DQN"


# ============================================================
# CRIA DIRETÓRIO
# ============================================================

echo "Criando diretório..."

mkdir -p "$TRAIN_DIR"


# ============================================================
# COPIA O PROJETO
# ============================================================

echo "Copiando arquivos..."

cp -r DQN "$TRAIN_DIR/"
cp requirements.txt "$TRAIN_DIR/"
cp Dockerfile "$TRAIN_DIR/"


# ============================================================
# ENTRA NO DIRETÓRIO
# ============================================================

cd "$TRAIN_DIR"


# ============================================================
# REMOVE CONTAINER ANTIGO
# ============================================================

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then

    echo "Removendo container antigo..."

    docker stop "$CONTAINER_NAME" 2>/dev/null || true

    docker rm "$CONTAINER_NAME"

fi


# ============================================================
# CRIA / ATUALIZA IMAGEM
# ============================================================

echo "Construindo imagem..."

docker build \
    -t "$IMAGE_NAME" \
    .


# ============================================================
# CRIA CONTAINER
# ============================================================

echo "Criando container..."

docker create \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    "$IMAGE_NAME"


# ============================================================
# INICIA CONTAINER
# ============================================================

echo "Iniciando IA..."

docker start "$CONTAINER_NAME"


# ============================================================
# INFORMAÇÕES
# ============================================================

echo ""
echo "=============================================="
echo "       SNAKE DQN INICIADO"
echo "=============================================="
echo ""
echo "Container: $CONTAINER_NAME"
echo "Imagem:    $IMAGE_NAME"
echo "Diretório: $TRAIN_DIR"
echo ""
echo "O container possui:"
echo "  restart: unless-stopped"
echo ""
echo "A IA inicia automaticamente pelo Dockerfile."
echo ""
echo "Ver treinamento:"
echo "docker logs -f $CONTAINER_NAME"
echo ""
echo "Ver status:"
echo "docker ps"
echo ""
echo "Parar manualmente:"
echo "docker stop $CONTAINER_NAME"
echo ""
echo "Iniciar novamente:"
echo "docker start $CONTAINER_NAME"
echo ""
echo "=============================================="