#!/bin/bash
# Resumo de todos os experimentos em RUNS_ROOT: estado do container, partidas e última avaliação.
RUNS_ROOT="${RUNS_ROOT:-$HOME/runs}"
printf "%-22s %-10s %8s %16s %10s %8s\n" "experimento" "container" "partida" "media±ic95" "melhor" "horas"
for dir in "$RUNS_ROOT"/*/; do
    name="$(basename "$dir")"
    state="$(docker inspect -f '{{.State.Status}}' "snake_$name" 2>/dev/null)" || state="-"
    last="$(tail -n 1 "$dir/logs/eval_log.csv" 2>/dev/null)"
    if [ -z "$last" ] || [[ "$last" == partida* ]]; then
        printf "%-22s %-10s %8s\n" "$name" "$state" "$(tail -n 1 "$dir/logs/training_log.csv" 2>/dev/null | cut -d, -f1)"
        continue
    fi
    IFS=, read -r partida media ic95 _ _ _ melhor _ _ _ _ horas _ <<< "$last"
    printf "%-22s %-10s %8s %16s %10s %8s\n" "$name" "$state" "$partida" "$media±$ic95" "$melhor" "$horas"
done
echo "Disco: $(du -sh "$RUNS_ROOT" 2>/dev/null | cut -f1) | Carga: $(cut -d' ' -f1-3 /proc/loadavg)"
