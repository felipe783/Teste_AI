import numpy as np
import torch

from Teste_IA.SnakeGameAI.Geral.snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet  


# Paramentros geneticos
MUTATION_RATE = 0.15 # % dos pesos que sofrem mutação em cada filho
MUTATION_STRENGTH = 0.30 # Desvio padrão do ruído gaussiano aplicado
MAX_STEPS_WITHOUT_FOOD = 100

class Agent:
    def __init__(self, model: Linear_QNet = None):
        self.model = model if model is not None else Linear_QNet(11,256,3) #Cria a Rede
        self.fitness = 0.0 # Desempenho do Individuo
        self.score = 0 

    # Estados
    def get_state(self, game):
            head = game.snake[0] # Posicao da Cabeca
            # Cria cada posicao ao redor da cabeca
            point_l = Point(head.x - BLOCK_SIZE, head.y)
            point_r = Point(head.x + BLOCK_SIZE, head.y)
            point_u = Point(head.x, head.y - BLOCK_SIZE)
            point_d = Point(head.x, head.y + BLOCK_SIZE)

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
            ]
            return np.array(state, dtype=np.float32) # Transforma o estado em um array, para ser usado na rede neural