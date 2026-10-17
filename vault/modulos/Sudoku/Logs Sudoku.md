# Sobre o Arquivo

O arquivo `Logs.py` é o **Responsável por Persistir as Métricas** de cada bateria em `results.json`, criando uma chave `teste_N` por execução.
# Funcionalidade

Converte os códigos de dificuldade e método em nomes, monta o dicionário de resultado e o acrescenta ao JSON existente:

````python
result = {
    "dificuldade": name,
    "tipo_resolucao": type,
    "jogos_gerados": numBoards,
    "jogos_resolvidos": solved,
    "jogos_impossiveis": impossible,
    "media_quadrantes_corretos": round(total_quadrants / numBoards, 2),
    "media_cobertura": round(total_coverage / numBoards, 2),
    "total_tentativas": total_attempts,
    "limite_tentativas_por_game": limit_attempts,
    "tempo_total": round(execution_time/60, 2),
    "desafio": challenge,
    "size": size
}
````

- `tempo_total` é gravado em **minutos**, enquanto o terminal mostra segundos.
- O arquivo é relativo ao diretório atual e é ignorado pelo Git (`*.json` no `.gitignore`).

> [!warning] Campos sobrescritos (confirmado em 2026-10-17)
> A função recebe `size` e `challenge`, mas as sobrescreve com `challenge = False` e `size = 9`. Execuções 16x16 ou Samurai ficam registradas como 9x9 tradicionais.
> Registros antigos usam a chave `limite_tentativas` e não têm `desafio`/`size`: o formato mudou ao longo do tempo.
