#!/bin/bash
# Lança um lote de experimentos em paralelo (um container de 1 CPU por experimento) com orçamento de partidas.
# Cada experimento tem a própria pasta em RUNS_ROOT/<nome> (Models/, logs/, config.json) e retoma se já existir.
#
#   bash aws/experiments.sh                      # lote padrão abaixo
#   AUTO_SHUTDOWN=1 bash aws/experiments.sh      # desliga a instância quando todos terminarem (economiza crédito)
#
# Edite EXPERIMENTS para o próximo lote. Formato: "nome|argumentos do agent.py".
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS_ROOT="${RUNS_ROOT:-$HOME/runs}"
GAMES="${GAMES:-40000}"

# Lote 1 (2026-11-05): ataca as duas causas prováveis do platô em ~100 pontos, todos do zero, mesma semente.
#  - base:     configuração atual (referência)
#  - safe:     exploração sem suicídio (com ε=0,02 o modelo de 62k faz ~31 pontos contra ~94 guloso)
#  - g099n3:   horizonte maior (γ=0,99 e retornos de 3 passos; L1 da auditoria)
#  - safe_g099n3: as duas mudanças juntas
EXPERIMENTS=(
    "base_s1|--seed 1"
    "safe_s1|--seed 1 --safe-explore"
    "g099n3_s1|--seed 1 --gamma 0.99 --n-step 3"
    "safe_g099n3_s1|--seed 1 --safe-explore --gamma 0.99 --n-step 3"
)

if [ "${#EXPERIMENTS[@]}" -gt "$(nproc)" ]; then
    echo "AVISO: ${#EXPERIMENTS[@]} experimentos para $(nproc) vCPU; cada um fica mais lento." >&2
fi

built=0
names=()
for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%%|*}"
    args="${entry#*|}"
    names+=("snake_$name")
    # shellcheck disable=SC2086  # os argumentos precisam ser separados por espaço
    SKIP_BUILD="$built" CONTAINER_NAME="snake_$name" BASE_DIR="$RUNS_ROOT/$name" RESTART_POLICY="on-failure:5" \
        bash run_docker.sh $args --games "$GAMES" | grep -E "^(Container|Dados|Args)"
    built=1
done

if [ "${AUTO_SHUTDOWN:-0}" = "1" ]; then
    # Espera todos os containers saírem (terminaram o orçamento ou falharam 5 vezes) e desliga a instância.
    # Instância parada não cobra CPU; o disco EBS continua cobrando (~US$ 0,08/GB-mês).
    nohup bash -c '
        while docker ps --format "{{.Names}}" | grep -qE "^('"$(IFS='|'; echo "${names[*]}")"')$"; do sleep 300; done
        sudo shutdown -h now' > "$RUNS_ROOT/auto_shutdown.log" 2>&1 &
    echo "Desligamento automático armado (log em $RUNS_ROOT/auto_shutdown.log)."
fi
echo "Acompanhe com: bash aws/status.sh"
