# Sobre o Arquivo

> [!warning] Versão local removida em 2026-10-29
> A pasta `SnakeGame/GA/` foi apagada no commit `0a73702`. O algoritmo descrito aqui continua só na cópia `SnakeGame/Server/Server_GA/GA/`, com as diferenças listadas em [[Server GA]], e essa cópia **não executa** no estado atual. Para ler o código antigo: `git show dfbc504:SnakeGame/GA/<arquivo>`.

O arquivo `train_genetic.py` é o **Loop Evolutivo**: avalia a população, registra métricas, salva o melhor e gera a próxima geração.

| Constante | Valor |
|---|---|
| `POPULATION_SIZE` | 80 |
| `GENERATIONS` | 500 por execução |
| `ELITE_FRACTION` | 0.15 → 12 elites |
| `TOURNAMENT_SIZE` | 5 |
# Funcionalidades

## tournament_selection

Sorteia 5 indivíduos e devolve o de maior fitness. Esse tamanho de torneio controla a pressão seletiva: um torneio maior dá mais chance aos melhores, com risco de perder diversidade.

## next_generation

````python
population.sort(key=lambda agent: agent.fitness, reverse=True)
n_elite = max(1, int(ELITE_FRACTION * POPULATION_SIZE))
new_population = [agent.clone() for agent in population[:n_elite]]

while len(new_population) < POPULATION_SIZE:
    parent_a = tournament_selection(population)
    parent_b = tournament_selection(population)
    child = Agent.crossover(parent_a, parent_b)
    child.mutate()
    new_population.append(child)
````

## train

1. Tenta carregar `best_genetic.pth`. Se existir, a população inicial é formada por **80 clones** do melhor e retoma a contagem de gerações.
2. Para cada geração: avalia todos (`play_and_evaluate`), calcula médias, salva o checkpoint se houver novo recorde, grava o log e atualiza o gráfico.
3. No `Ctrl+C`, salva o melhor da geração atual.

> [!note] Retomada sem diversidade
> Ao retomar, os 80 indivíduos da primeira geração são idênticos e só se diferenciam a partir da segunda (crossover + mutação). Na prática, a primeira geração retomada reavalia o mesmo genoma 80 vezes, variando apenas a posição aleatória da comida.
