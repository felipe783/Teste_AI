import random
import time
from SnakeGameAI.GA.agent import Agent
from Teste_IA.SnakeGameAI.GA.Helper import plot


POPULATION_SIZE = 80 
GENERATIONS = 500
ELITE_FRACTION = 0.10
TOURNAMENT_SIZE = 5

# Selecao

def tournament_selection(population): 
    # Pega 5 individuos aleatorios da populacao e retorna o melhor
    competitors = random.sample(population, TOURNAMENT_SIZE)
    return max(competitors, key=lambda agent: agent.fitness)

def next_generation(population):
    # Melhor para o pior
    population.sort(key=lambda agent: agent.fitness, reverse=True)

    # Numero de individuos que serao mantidos na proxima geracao
    # Esta logica usa o Elitismo para garantir que a boa genetica nao seja perdidada
    n_elite = max(1, int(ELITE_FRACTION * POPULATION_SIZE))
    new_population = [agent.clone() for agent in population[:n_elite]]

    while len(new_population) < POPULATION_SIZE:
        # Pega 2 pais para o Crossover
        parent_a = tournament_selection(population)
        parent_b = tournament_selection(population)
        child = Agent.crossover(parent_a, parent_b)
        child.mutate() 
        new_population.append(child) # 8 Elites e 72 Filhos

    return new_population

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
        print(
            f"Genética anterior carregada "
            f"(geração {start_gen}, recorde {record})"
        )
    generation = start_gen

    try:
        for generation in range(start_gen, start_gen + GENERATIONS):

            # Inicio Geracao
            gen_start_time = time.time()
            
            gen_best_score = 0

            for agent in population:
                _, score = agent.play_and_evaluate()
                gen_best_score = max(gen_best_score, score)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score /  len(plot_scores)
                plot_mean_scores.append(mean_score)

            best_in_gen = max(population, key=lambda a: a.fitness)

            # Medias
            average_score = (sum(agent.score for agent in population)/ len(population))

            average_fitness = (sum(agent.fitness for agent in population)/ len(population))

            elapsed_time = (time.time() - gen_start_time)

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

            best_in_gen.save_generation_log(
                            generation=generation,
                            best_score=best_in_gen.score,
                            best_fitness=best_in_gen.fitness,
                            average_score=average_score,
                            average_fitness=average_fitness,
                            population_size=len(population),
                            mutation_rate=best_in_gen.mutation_rate,
                            mutation_strength=best_in_gen.mutation_strength,
                            elapsed_time=elapsed_time,
                            population=population
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