"""
Agent para o treinamento por gerações (Algoritmo Genético).

Diferente do Agent do DQN, este Agent NÃO tem memory, trainer ou epsilon.
Cada instância representa um indivíduo (uma cobra) cujo "aprendizado"
acontece entre gerações, via seleção + crossover + mutação dos pesos
da rede — não por gradiente.
"""

import random
import numpy as np
import torch

from SnakeGameAI.DQN.snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from SnakeGameAI.GA.model import Linear_QNet

# ----------------------- Hiperparâmetros genéticos -----------------------

MUTATION_RATE = 0.15        # % dos pesos que sofrem mutação em cada filho
MUTATION_STRENGTH = 0.30    # desvio padrão do ruído gaussiano aplicado
MAX_STEPS_WITHOUT_FOOD = 100  # trava cobras que ficam girando sem comer


class Agent:
    def __init__(self, model: Linear_QNet = None):
        # mesma arquitetura do DQN, mas sem trainer/otimizador associado
        self.model = model if model is not None else Linear_QNet(11, 256, 3)
        self.fitness = 0.0
        self.score = 0

    # ----------------------- Estado (idêntico ao DQN) -----------------------

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

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
        ]
        return np.array(state, dtype=np.float32)

    # ----------------------- Ação (sem epsilon — sempre greedy) -----------------------

    @torch.no_grad()
    def get_action(self, state):
        # Sem exploração aleatória aqui: a diversidade vem da população,
        # não de escolhas aleatórias dentro da partida.
        state0 = torch.tensor(state, dtype=torch.float32)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

    # ----------------------- Avaliação (substitui train_short/long_memory) -----------------------

    def play_and_evaluate(self):
        """Roda uma partida completa headless e calcula o fitness do indivíduo."""
        game = SnakeGameAI()
        steps = 0
        steps_since_food = 0
        max_steps_since_food = MAX_STEPS_WITHOUT_FOOD

        while True:
            state = self.get_state(game)
            action = self.get_action(state)
            reward, done, score = game.play_step(action)

            steps += 1
            if reward > 0:
                steps_since_food = 0
                max_steps_since_food += MAX_STEPS_WITHOUT_FOOD
            else:
                steps_since_food += 1

            if steps_since_food > max_steps_since_food:
                done = True

            if done:
                self.score = score
                self.fitness = (score ** 2) * 100 + steps
                return self.fitness, score

    # ----------------------- Manipulação de pesos (delega ao model) -----------------------
    # get_flat_weights/set_flat_weights agora vivem em Linear_QNet (model.py) —
    # são sobre a estrutura da rede, não sobre o comportamento do agente.

    def clone(self):
        child = Agent(model=self.model.clone())
        return child

    # ----------------------- Operadores genéticos -----------------------

    @staticmethod
    def crossover(parent_a: "Agent", parent_b: "Agent") -> "Agent":
        wa = parent_a.model.get_flat_weights()
        wb = parent_b.model.get_flat_weights()
        mask = torch.rand_like(wa) < 0.5
        child_weights = torch.where(mask, wa, wb)
        child = Agent()
        child.model.set_flat_weights(child_weights)
        return child

    def mutate(self):
        weights = self.model.get_flat_weights()
        mask = torch.rand_like(weights) < MUTATION_RATE
        noise = torch.randn_like(weights) * MUTATION_STRENGTH
        weights[mask] += noise[mask]
        self.model.set_flat_weights(weights)

    # ----------------------- Checkpoint (delega ao model) -----------------------

    def save_checkpoint(self, generation, record):
        self.model.save_checkpoint(generation, record)

    @staticmethod
    def load_checkpoint():
        model = Linear_QNet(11, 256, 3)
        generation, record = model.load_checkpoint()
        if generation == 0 and record == 0:
            return None, 0, 0
        agent = Agent(model=model)
        return agent, generation, record
