#!/bin/bash
# Mede passos/s com N treinos em paralelo (padrão: um por vCPU) para confirmar a instância antes de gastar horas.
# Treinos do zero em pastas temporárias, apagadas no fim; não toca em nenhuma execução real.
#   bash aws/benchmark.sh            # N = nproc, 180 s
#   N=2 SECONDS_RUN=300 bash aws/benchmark.sh
set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-$(nproc)}"
SECONDS_RUN="${SECONDS_RUN:-180}"
TMP="$(mktemp -d)"
docker build -q -t snake_dqn . >/dev/null

for i in $(seq "$N"); do
    mkdir -p "$TMP/b$i"
    docker run -d --rm --name "snake_bench_$i" --cpus 1 --user "$(id -u):$(id -g)" -v "$TMP/b$i:/data" \
        snake_dqn --seed "$i" --checkpoint-every 50 --eval-every 1000000 --no-save-replay >/dev/null
done
echo "$N treinos rodando por ${SECONDS_RUN}s..."
sleep "$SECONDS_RUN"

total=0
for i in $(seq "$N"); do
    rate="$(docker logs "snake_bench_$i" 2>&1 | grep '^\[status\]' | tail -1 | sed -E 's/.*passos\/s=([0-9]+).*/\1/')"
    echo "treino $i: ${rate:-?} passos/s"
    total=$((total + ${rate:-0}))
    docker stop -t 120 "snake_bench_$i" >/dev/null
done
echo "TOTAL: $total passos/s com $N treinos em $(nproc) vCPU"
echo "Referência (notebook Ryzen AI 5 340, 2026-11-05): ~2 400 passos/s por treino; 4 em paralelo: ~1 850 cada."
rm -rf "$TMP"
