import numpy as np
import torch

from Teste_IA.SnakeGameAI.Geral.snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import DEVICE, Linear_QNet  


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

    #Acao
    @torch.no_grad() # Nao precisa calcular o gradiente
    def get_action(self, state):  
        # Um tensor pode ter qualquer quantidade de dimensao  
        state0 = torch.tensor(state,  dtype=torch.float32).to(DEVICE) # Transforma o array pro tensor
        prediction = self.model(state0) # Oq retornar da Rede
        move = torch.argmax(prediction).item() # Pega o valor max, e transforma em um numero inteiro
        final_move = [0, 0, 0] 
        final_move[move] = 1 
        return final_move

    #Avaliacao

    def play_and_evaluate(self):

        # Ira rodar uma partida headless e ver o desemepnho
        game = SnakeGameAI() 
        steps = 0
        steps_since_food = 0
        max_steps_without_food = MAX_STEPS_WITHOUT_FOOD

        while True:
            state = self.get_state(game)
            action = self.get_action(state)
            reward, done, score = game.play_step(action)

            steps += 1

            if reward > 0:
                steps_since_food = 0
                max_steps_without_food = MAX_STEPS_WITHOUT_FOOD
            else:
                steps_since_food += 1
            if steps_since_food > max_steps_without_food:
                done = True
            if done:
                self.score = score
                self.fitness = (score ** 2) * 100 + steps
                return self.fitness,score