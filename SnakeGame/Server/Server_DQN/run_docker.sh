#!/bin/bash
# Constrói a imagem e (re)cria um container de treino. O treino retoma do que houver em BASE_DIR.
#
#   bash run_docker.sh                                   # treino padrão, dados em BASE_DIR
#   CONTAINER_NAME=gamma099 BASE_DIR=~/runs/gamma099 bash run_docker.sh --gamma 0.99 --seed 1 --games 20000
#
# Variáveis (todas opcionais):
#   BASE_DIR        pasta do host com Models/, logs/ e config.json (padrão: a do servidor caseiro)
#   CONTAINER_NAME  nome do container (padrão Snake_DQN); um por experimento
#   RESTART_POLICY  unless-stopped (padrão: treino sem fim) ou on-failure:5 (com --games, para não reiniciar ao terminar)
#   CPUS            limite de CPUs do container (padrão 1: cada treino usa 1 núcleo)
#   SKIP_BUILD=1    não reconstrói a imagem
# Argumentos depois do script vão para o DQN/agent.py (ver python DQN/agent.py --help).

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-snake_dqn}"
CONTAINER_NAME="${CONTAINER_NAME:-Snake_DQN}"
BASE_DIR="${BASE_DIR:-/home/felipe/Teste_IA/Server_DQN}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"
CPUS="${CPUS:-1}"

cd "$(dirname "$0")"

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "Construindo imagem..."
    COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo desconhecido)"
    if [ -n "$(git status --porcelain -- . 2>/dev/null)" ]; then COMMIT="${COMMIT}-dirty"; fi
    docker build --build-arg GIT_COMMIT="$COMMIT" -t "$IMAGE_NAME" .
fi

echo "Verificando a pasta de dados no host: $BASE_DIR"
mkdir -p "$BASE_DIR/Models" "$BASE_DIR/logs"
# O container roda com o usuário do host. Volumes criados por versões antigas (que rodavam como root)
# precisam mudar de dono uma vez, senão o salvamento atômico falha por permissão.
if [ -n "$(find "$BASE_DIR" ! -user "$(id -u)" -print -quit)" ]; then
    echo "ERRO: há arquivos em $BASE_DIR que não são do usuário $(id -un). Rode uma vez:"
    echo "  sudo chown -R $(id -u):$(id -g) \"$BASE_DIR\""
    exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removendo container antigo (o treino salva ao receber SIGTERM)..."
    docker stop -t 120 "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm "$CONTAINER_NAME" >/dev/null
fi

echo "Criando container..."
# --stop-timeout: tempo para salvar ao receber SIGTERM (docker stop, desligamento da instância)
# --user: os arquivos no volume ficam com o dono do host, não root
docker create \
    --name "$CONTAINER_NAME" \
    --restart "$RESTART_POLICY" \
    --stop-timeout 120 \
    --cpus "$CPUS" \
    --user "$(id -u):$(id -g)" \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    -v "$BASE_DIR:/data" \
    "$IMAGE_NAME" "$@" >/dev/null

docker start "$CONTAINER_NAME" >/dev/null

echo ""
echo "=============================================="
echo "       SNAKE DQN INICIADO"
echo "=============================================="
echo "Container: $CONTAINER_NAME  (restart: $RESTART_POLICY, cpus: $CPUS)"
echo "Imagem:    $IMAGE_NAME"
echo "Dados:     $BASE_DIR -> /data  (Models/, logs/, config.json)"
echo "Args:      ${*:-(padrão / config.json salvo)}"
echo ""
echo "Acompanhar:  docker logs -f $CONTAINER_NAME | grep -v '^Partida'"
echo "Avaliações:  column -s, -t < $BASE_DIR/logs/eval_log.csv | tail"
echo "Parar:       docker stop -t 120 $CONTAINER_NAME   (salva antes de sair)"
