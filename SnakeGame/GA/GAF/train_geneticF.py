"""
Loop de treino por gerações (Algoritmo Genético) para a Snake AI.

Usa a classe Agent de agent.py — cada Agent é um indivíduo da população.
Este arquivo cuida só da orquestração: criar a população, avaliar,
selecionar, gerar a próxima geração e plotar o progresso.

Uso (a partir da raiz do projeto, acima de SnakeGameAI/):
    python -m SnakeGameAI.GA.train_genetic
"""

import random

from SnakeGameAI.GA.agent import Agent
from Teste_IA.SnakeGameAI.DQN.Helper import plot

# ----------------------- Hiperparâmetros -----------------------

POPULATION_SIZE = 80
GENERATIONS = 500
ELITE_FRACTION = 0.10
TOURNAMENT_SIZE = 5
# Caminho do checkpoint é gerenciado internamente por model.py (MODEL_DIR)


# ----------------------- Seleção -----------------------

def tournament_selection(population):
    competitors = random.sample(population, TOURNAMENT_SIZE)
    return max(competitors, key=lambda a: a.fitness)


def next_generation(population):
    population.sort(key=lambda a: a.fitness, reverse=True)

    n_elite = max(1, int(ELITE_FRACTION * POPULATION_SIZE))
    new_population = [a.clone() for a in population[:n_elite]]

    while len(new_population) < POPULATION_SIZE:
        parent_a = tournament_selection(population)
        parent_b = tournament_selection(population)
        child = Agent.crossover(parent_a, parent_b)
        child.mutate()
        new_population.append(child)

    return new_population


# ----------------------- Loop principal -----------------------

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    start_gen = 0

    seed_agent, start_gen, record = Agent.load_checkpoint()
    population = [Agent() for _ in range(POPULATION_SIZE)]
    if seed_agent is not None:
        population[0] = seed_agent

    generation = start_gen

    try:
        for generation in range(start_gen, start_gen + GENERATIONS):
            gen_best_score = 0

            for agent in population:
                _, score = agent.play_and_evaluate()
                gen_best_score = max(gen_best_score, score)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / len(plot_scores)
                plot_mean_scores.append(mean_score)

            best_in_gen = max(population, key=lambda a: a.fitness)

            if best_in_gen.score > record:
                record = best_in_gen.score
                print(f'Novo recorde: {record} (geração {generation})')
                best_in_gen.save_checkpoint(generation, record)

            print(
                'Geração:', generation,
                'Melhor da geração:', gen_best_score,
                'Recorde geral:', record,
                'Fitness top:', round(best_in_gen.fitness, 1)
            )

            plot(plot_scores, plot_mean_scores)

            population = next_generation(population)

    except KeyboardInterrupt:
        print("\nTreinamento interrompido.")
        best_in_gen = max(population, key=lambda a: a.fitness)
        best_in_gen.save_checkpoint(generation, record)
        print("Melhor Agent salvo!")
        print("Pode fechar o programa.")


if __name__ == "__main__":
    train()
