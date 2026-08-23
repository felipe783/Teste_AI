import torch 
import random 
import numpy as np
from collections import deque
from snake_gameai import SnakeGameAI,Direction,Point,BLOCK_SIZE
from model import Linear_QNet,QTrainer
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):

        self.n_game = 0
        self.epsilon = 80
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(11, 256, 3)

        self.trainer = QTrainer(
            self.model,
            lr=LR,
            gamma=self.gamma
        )

        # Carrega checkpoint se existir
        self.n_game, self.record, self.epsilon = \
            self.model.load_checkpoint(
                self.trainer.optimizer
            )

        print(
            f'Continuando treinamento: '
            f'Game={self.n_game}, '
            f'Record={self.record}'
        )
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n) 
        # self.model.to('cuda')   
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)         
        # TODO: model,trainer

    # state (11 Values)
    #[ danger straight, danger right, danger left,
    #   
    # direction left, direction right,
    # direction up, direction down
    # 
    # food left,food right,
    # food up, food down]
    def get_state(self,game):
        head = game.snake[0]
        point_l=Point(head.x - BLOCK_SIZE, head.y)
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d))or
            (dir_l and game.is_collision(point_l))or
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
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            game.food.x < game.head.x, # food is in left
            game.food.x > game.head.x, # food is in right
            game.food.y < game.head.y, # food is up
            game.food.y > game.head.y  # food is down
        ]
        return np.array(state,dtype=int)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.n_game
        #self.epsilon = max(0.01, 20 - self.n_game * 0.2)
        final_move = [0,0,0]
        if(random.randint(0,200)<self.epsilon):
            move = random.randint(0,2)
            final_move[move]=1
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)# prediction by model 
            move = torch.argmax(prediction).item()
            final_move[move]=1 
        return final_move

def train():

    plot_scores = []
    plot_mean_scores = []

    total_score = 0

    agent = Agent()
    game = SnakeGameAI()

    try:

        while True:

            # Estado atual
            state_old = agent.get_state(game)

            # Escolhe movimento
            final_move = agent.get_action(state_old)

            # Executa movimento
            reward, done, score = game.play_step(final_move)

            # Novo estado
            state_new = agent.get_state(game)

            # Treinamento curto
            agent.train_short_memory(
                state_old,
                final_move,
                reward,
                state_new,
                done
            )

            # Guarda memória
            agent.remember(
                state_old,
                final_move,
                reward,
                state_new,
                done
            )

            if done:
                # Gravar partida
                if score >= agent.record - 5:
                    game.save_video(score, agent.record)                

                game.reset()
                agent.n_game += 1
                agent.train_long_memory()

                # Novo recorde
                if score > agent.record:
                    agent.record = score
                    print(f'Novo recorde: {agent.record}')
                print(
                    'Game:', agent.n_game,
                    'Score:', score,
                    'Record:', agent.record,
                    'Epsilon:', agent.epsilon
                )

                # Salva checkpoint a cada jogo
                agent.model.save_checkpoint(
                    agent.trainer.optimizer,
                    agent.n_game,
                    agent.record,
                    agent.epsilon
                )

                plot_scores.append(score)
                total_score += score
                mean_score = (total_score / len(plot_scores))
                plot_mean_scores.append(mean_score)

                plot(plot_scores,plot_mean_scores)

    except KeyboardInterrupt:

        print("\nTreinamento interrompido.")

        agent.model.save_checkpoint(
            agent.trainer.optimizer,
            agent.n_game,
            agent.record,
            agent.epsilon
        )

        print("Checkpoint salvo!")
        print("Pode fechar o programa.")

if(__name__=="__main__"):
    train()