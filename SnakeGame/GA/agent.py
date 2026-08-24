import json
import math
import numpy as np
import torch
import os
from snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import DEVICE, Linear_QNET

MUTATION_RATE = 0.15       # % dos pesos que sofrem mutação em cada filho
MUTATION_STRENGTH = 0.30   # Desvio padrão do ruído gaussiano aplicado
MAX_STEPS_WITHOUT_FOOD = 100

STATE_SIZE = 15

# Limites de segurança para os parâmetros auto-adaptativos —
# sem isso, mutation_rate/strength poderiam "explodir" ou
MIN_MUTATION_RATE = 0.01
MAX_MUTATION_RATE = 0.9
MIN_MUTATION_STRENGTH = 0.01
MAX_MUTATION_STRENGTH = 2.0

# Tau: controla o quanto mutation_rate/strength podem mudar por geração. 
SELF_ADAPT_TAU = 0.1

LOG_DIR = "logs"

class Agent:
    def __init__(
        self,
        model: Linear_QNET = None,
        mutation_rate: float = None,
        mutation_strength: float = None
    ):

        self.model = model if model is not None else Linear_QNET(STATE_SIZE, 256, 3)  # Cria a Rede
        self.fitness = 0.0  # Desempenho do Individuo
        self.score = 0

        # Parâmetros de estratégia PRÓPRIOS do indivíduo (mutação auto-adaptativa)
        self.mutation_rate = (
            mutation_rate if mutation_rate is not None else MUTATION_RATE
        )
        self.mutation_strength = (
            mutation_strength if mutation_strength is not None else MUTATION_STRENGTH
        )

    def get_state(self, game):

        head = game.snake[0]  # Posicao da Cabeca

        # Cria cada posicao ao redor da cabeca
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Ver 2 Blocos pra frente — visão de perigo além do
        # imediato, ajuda a evitar que a cobra se
        # autoencurrale conforme cresce.
        point_l2 = Point(head.x - (2 * BLOCK_SIZE), head.y)
        point_r2 = Point(head.x + (2 * BLOCK_SIZE), head.y)
        point_u2 = Point(head.x, head.y - (2 * BLOCK_SIZE))
        point_d2 = Point(head.x, head.y + (2 * BLOCK_SIZE))

        # Descobrir a direcao
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Verificar se existe uma colicao para o ponto que a cabeca esta indo
        state = [
            # Danger Straight
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger Left
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)),

            # Move Direction
            dir_l, dir_r, dir_u, dir_d,

            # Food Location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,

            # Perigo 2 Blocos
            (
                (dir_u and game.is_collision(point_u2)) or
                (dir_d and game.is_collision(point_d2)) or
                (dir_l and game.is_collision(point_l2)) or
                (dir_r and game.is_collision(point_r2))
            ),

            # Perigo a direita (2 blocos)
            (
                (dir_u and game.is_collision(point_r2)) or
                (dir_d and game.is_collision(point_l2)) or
                (dir_l and game.is_collision(point_u2)) or
                (dir_r and game.is_collision(point_d2))
            ),

            # Perigo a esquerda (2 blocos)
            (
                (dir_u and game.is_collision(point_l2)) or
                (dir_d and game.is_collision(point_r2)) or
                (dir_l and game.is_collision(point_d2)) or
                (dir_r and game.is_collision(point_u2))
            ),

            # Tamanho da Cobra
            len(game.snake) / ((game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE)),
        ]

        return np.array(state, dtype=np.float32)  # Transforma o estado em um array, para ser usado na rede neural

    @torch.no_grad()  # Nao precisa calcular o gradiente
    def get_action(self, state):

        # Um tensor pode ter qualquer quantidade de dimensao
        state0 = torch.tensor(state, dtype=torch.float32).to(DEVICE)  # Transforma o array pro tensor
        prediction = self.model(state0)  # Oq retornar da Rede
        move = torch.argmax(prediction).item()  # Pega o valor max, e transforma em um numero inteiro
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

    def play_and_evaluate(self):

        # Ira rodar uma partida headless e ver o desemepnho
        game = SnakeGameAI()  # Nova partida
        steps = 0
        steps_since_food = 0
        max_steps_without_food = MAX_STEPS_WITHOUT_FOOD

        while True:

            state = self.get_state(game)
            action = self.get_action(state)  # Vai pra rede neural
            reward, done, score = game.play_step(action)

            steps += 1

            if reward > 0:  # Se comeu a comida
                steps_since_food = 0
                max_steps_without_food = MAX_STEPS_WITHOUT_FOOD
            else:
                steps_since_food += 1

            if steps_since_food > max_steps_without_food:  # Evitar LOOP
                done = True

            if done:  # Fim game
                self.score = score
                self.fitness = (score ** 2) * 100 + steps  # Dando mais importancia para o score, mas tambem para os passos
                return self.fitness, score

    # Tem os mesmo pesos mas sao objetos diferentes
    def clone(self):

        # O clone herda os parâmetros de estratégia do pai
        child = Agent(
            model=self.model.clone(),
            mutation_rate=self.mutation_rate,
            mutation_strength=self.mutation_strength
        )
        return child

    @staticmethod
    def crossover(parent_a: "Agent", parent_b: "Agent") -> "Agent":

        # Pega os Pesos dos pais
        wa = parent_a.model.get_flat_weights()
        wb = parent_b.model.get_flat_weights()

        # Cria numeros aleatorios entre 1 e 0, com o tamanho de "wa", e transforma em Bool
        mask = torch.rand_like(wa) < 0.5

        # Se for true pega A se for false pega B
        child_weights = torch.where(mask, wa, wb)

        child_mutation_rate = (parent_a.mutation_rate + parent_b.mutation_rate) / 2
        child_mutation_strength = (parent_a.mutation_strength + parent_b.mutation_strength) / 2

        child = Agent(
            mutation_rate=child_mutation_rate,
            mutation_strength=child_mutation_strength
        )

        child.model.set_flat_weights(child_weights)  # Pega A + B e coloca na rede

        return child

    def mutate(self):

        self.mutation_strength *= math.exp(
            SELF_ADAPT_TAU * np.random.randn()
        )

        self.mutation_strength = float(
            np.clip(
                self.mutation_strength,
                MIN_MUTATION_STRENGTH,
                MAX_MUTATION_STRENGTH
            )
        )

        self.mutation_rate *= math.exp(SELF_ADAPT_TAU * np.random.randn())
        self.mutation_rate = float(
            np.clip(
                self.mutation_rate,
                MIN_MUTATION_RATE,
                MAX_MUTATION_RATE
            )
        )

        # Mutacao dos Pesos
        weights = self.model.get_flat_weights()

        # Decide quais pesos irao sofrer mutacao
        mask = torch.rand_like(weights) < self.mutation_rate

        # Gera um ruido aleatorio
        noise = torch.randn_like(weights) * self.mutation_strength

        weights[mask] += noise[mask]  # Aplica o Ruido nos pesos que foram selecionados

        self.model.set_flat_weights(weights)

    def save_checkpoint(self, generation, record):
        self.model.save_checkpoint(generation, record)

    @staticmethod
    def load_checkpoint():
        model = Linear_QNET(STATE_SIZE, 256, 3)
        generation, record = model.load_checkpoint()

        if generation == 0 and record == 0:
            return None, 0, 0

        agent = Agent(model=model)
        return agent, generation, record

    def save_generation_log(
        self,
        generation,
        best_score,
        best_fitness,
        average_score,
        average_fitness,
        population_size,
        mutation_rate,
        mutation_strength,
        elapsed_time,
        population
    ):

        os.makedirs(LOG_DIR, exist_ok=True)
        filepath = os.path.join(LOG_DIR, "history.json")

        if os.path.exists(filepath):

            with open(filepath, "r", encoding="utf-8") as file:
                history = json.load(file)

        else:

            history = []

        # Desempenho de cada indivíduo
        individuals_data = []

        for i, agent in enumerate(population):

            individuals_data.append({
                "id": i,
                "score": agent.score,
                "fitness": agent.fitness,
                "mutation_rate": agent.mutation_rate,
                "mutation_strength": agent.mutation_strength
            })

        # Dados da geração
        generation_data = {
            "generation": generation,
            "best_score": best_score,
            "best_fitness": best_fitness,
            "average_score": average_score,
            "average_fitness": average_fitness,
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "mutation_strength": mutation_strength,
            "elapsed_time": elapsed_time,
            "individuals": individuals_data
        }

        history.append(generation_data)

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=4)