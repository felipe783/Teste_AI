import os
import csv
import torch
import random
import numpy as np

from collections import deque
from snake_gameai import (SnakeGameAI,Direction,Point,BLOCK_SIZE)
from model import (Linear_QNet,QTrainer,DEVICE)

MAX_MEMORY = 100_000 # Replay Memory, fica amazenado 100.000 experiências.
BATCH_SIZE = 1000 #  Pega 1.000 experiências aleatórias da memória e usa para atualizar a rede neural.
LR = 0.001 # Learning Rate, Controla o tamanho dos ajustes feitos nos pesos da rede neural.
STATE_SIZE = 15  # 15 Estados
TARGET_UPDATE_FREQ = 1000 

# DIR
MODEL_DIR = "Models"
MODEL_PATH = os.path.join(MODEL_DIR,"model.pth")
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR,"training_log.csv")

# Limite CSV
LOG_MAX_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB

# Log
class TrainingLogger:

    def __init__(self):
        self.total_games = 0
        self.record = 0

        os.makedirs(LOG_DIR,exist_ok=True)

        if os.path.exists(LOG_FILE):
            with open(LOG_FILE,"r",newline="",encoding="utf-8") as file:

                reader = csv.DictReader(file)
                for row in reader:
                    try:
                        self.total_games = max(self.total_games,int(row["partida"]))
                        self.record = max(self.record,int(row["max"]))
                    except (ValueError,KeyError):
                        continue

    # Sincornizar com o checkpoint 
    def sync_with_agent(self, agent):

        # O checkpoint possui a contagem mais confiável do treinamento.
        self.total_games = max(self.total_games,agent.n_game)
        self.record = max(self.record,agent.record)

    # Rotação Log
    def _rotate_if_needed(self):

        if not os.path.exists(LOG_FILE):
            return

        size = os.path.getsize(LOG_FILE)
        if size >= LOG_MAX_SIZE_BYTES:
            os.remove(LOG_FILE)
            print(
                f"Log de treino excedeu "
                f"{LOG_MAX_SIZE_BYTES / (1024**3):.1f} GB "
                f"({size / (1024**3):.2f} GB) — arquivo "
                f"zerado e recomeçado do zero.",
                flush=True
            )

    # Log partida
    def log_game(self, score):

        self.total_games += 1
        self.record = max(self.record,score)
        self._rotate_if_needed()

        file_exists = os.path.exists(LOG_FILE)

        with open(LOG_FILE,"a",newline="",encoding="utf-8") as file:

            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "partida",
                    "score",
                    "max",
                    "total_partidas"
                ])

            writer.writerow([
                self.total_games,
                score,
                self.record,
                self.total_games
            ])

        print(
            f"Partida: {self.total_games} | "
            f"Score: {score} | "
            f"Max: {self.record} | "
            f"Total: {self.total_games}",
            flush=True
        )

# Agent 
class Agent:

    def __init__(self):

        self.n_game = 0
        self.record = 0
        self.epsilon = 80 # Server pra exploração
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(STATE_SIZE,256,3)

        # Train com Target Network
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma,target_update_freq=TARGET_UPDATE_FREQ) # Treinar a Rede
        self.load_checkpoint()

    # Carregar checkpoint
    def load_checkpoint(self):
        if not os.path.exists(MODEL_PATH):
            print(
                f"Nenhum checkpoint encontrado em: "
                f"{MODEL_PATH}",
                flush=True
            )
            return
        print(
            f"Carregando checkpoint: "
            f"{MODEL_PATH}",
            flush=True
        )

        # Load Checkpoint
        checkpoint = torch.load(MODEL_PATH,map_location=DEVICE)

        if (isinstance(checkpoint, dict)and"model_state_dict" in checkpoint):

            self.model.load_state_dict(checkpoint["model_state_dict"])

            if "optimizer_state_dict" in checkpoint:
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "target_model_state_dict" in checkpoint:
                self.trainer.load_target_state_dict(checkpoint["target_model_state_dict"])

            else:
                self.trainer.update_target_model()

            self.n_game = checkpoint.get("n_game",0)
            self.record = checkpoint.get("record",0)

        # State Antigo
        else:
            self.model.load_state_dict(checkpoint)
            self.trainer.update_target_model()

        self.model.to(DEVICE)

        print("Checkpoint carregado!",flush=True)
        print(f"Partida inicial: {self.n_game}",flush=True)
        print(f"Recorde inicial: {self.record}",flush=True)

    # Checkpoint Save
    # So é chamado quando um Record é batido
    def save_checkpoint(self):

        os.makedirs(MODEL_DIR,exist_ok=True)
        checkpoint = {
            "model_state_dict":self.model.state_dict(),
            "target_model_state_dict":self.trainer.target_model.state_dict(),
            "optimizer_state_dict":self.trainer.optimizer.state_dict(),
            "n_game":self.n_game,
            "record":self.record
        }

        torch.save(checkpoint,MODEL_PATH)
        print(
            f"Checkpoint salvo em: "
            f"{MODEL_PATH} (novo recorde: {self.record})",
            flush=True
        )

    # Estados
    def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - BLOCK_SIZE,head.y)
        point_r = Point(head.x + BLOCK_SIZE,head.y)
        point_u = Point(head.x,head.y - BLOCK_SIZE)
        point_d = Point(head.x,head.y + BLOCK_SIZE)

        # Ver 2 Blocos pra frente
        point_l2 = Point(head.x - (2 * BLOCK_SIZE),head.y)
        point_r2 = Point(head.x + (2 * BLOCK_SIZE),head.y)
        point_u2 = Point(head.x,head.y - (2 * BLOCK_SIZE))
        point_d2 = Point(head.x,head.y + (2 * BLOCK_SIZE))

        dir_l = (game.direction == Direction.LEFT)
        dir_r = (game.direction == Direction.RIGHT)
        dir_u = (game.direction == Direction.UP)
        dir_d = (game.direction == Direction.DOWN)

        state = [
            # Perigo a frente
            (
                (dir_u and game.is_collision(point_u)) or
                (dir_d and game.is_collision(point_d)) or
                (dir_l and game.is_collision(point_l)) or
                (dir_r and game.is_collision(point_r))
            ),

            # Perigo a direita
            (
                (dir_u and game.is_collision(point_r)) or
                (dir_d and game.is_collision(point_l)) or
                (dir_l and game.is_collision(point_u)) or
                (dir_r and game.is_collision(point_d))
            ),

            # Perigo a Esquerda
            (
                (dir_u and game.is_collision(point_l)) or
                (dir_d and game.is_collision(point_r)) or
                (dir_l and game.is_collision(point_d)) or
                (dir_r and game.is_collision(point_u))
            ),

            # Direcao
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Comida
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,

            # Perigo 2 Blocos
            (
                (dir_u and game.is_collision(point_u2)) or
                (dir_d and game.is_collision(point_d2)) or
                (dir_l and game.is_collision(point_l2)) or
                (dir_r and game.is_collision(point_r2))
            ),

            # Perigo a direita
            (
                (dir_u and game.is_collision(point_r2)) or
                (dir_d and game.is_collision(point_l2)) or
                (dir_l and game.is_collision(point_u2)) or
                (dir_r and game.is_collision(point_d2))
            ),

            # Perigo a esquerda
            (
                (dir_u and game.is_collision(point_l2)) or
                (dir_d and game.is_collision(point_r2)) or
                (dir_l and game.is_collision(point_d2)) or
                (dir_r and game.is_collision(point_u2))
            ),

            # Tamanho da Cobra
            len(game.snake) / ((game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE))
        ]
        return np.array(state,dtype=np.float32)

    # Guarda a Experiencia 
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    # Long Memory
    def train_long_memory(self): # Treina com as 1000 Experiencias aleatorias 
        if len(self.memory) < BATCH_SIZE:
            return
        mini_sample = random.sample(self.memory,BATCH_SIZE)
        (states,actions,rewards,next_states,dones) = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    # Shot Memory
    def train_short_memory(self,state,action,reward,next_state,done): # Treina oq acabou de acontecer
        self.trainer.train_step(state,action,reward,next_state,done)

    # Tomada de Ação
    def get_action(self, state):  
        self.epsilon = max(5,80 - self.n_game)
        final_move = [0,0,0]

        # Exploração
        # Movimento aleatorio
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)

        # Modelo
        # Movimento calculado pela Rede
        else:
            state0 = torch.as_tensor(np.array(state),dtype=torch.float32,device=DEVICE)
            with torch.no_grad(): # Apenas escolhendo uma Ação, não esta treinando
                prediction = self.model(state0)

            move = torch.argmax(prediction).item() 
        final_move[move] = 1
        return final_move

# Treinamento
def train():

    agent = Agent()
    logger = TrainingLogger()
    logger.sync_with_agent(agent)
    game = SnakeGameAI()

    print("==============================================",flush=True)
    print("       TREINAMENTO DQN INICIADO",flush=True)
    print(f"Modelo: {MODEL_PATH}",flush=True)
    print(f"Partida inicial: {agent.n_game}",flush=True)
    print(f"Recorde inicial: {agent.record}",flush=True)
    print(f"Logger inicial: {logger.total_games}",flush=True)
    print("==============================================",flush=True)
    try:
        while True:
            # Estado Atual
            state_old = agent.get_state(game)

            # Escolher ação
            final_move = agent.get_action(state_old)

            # Executa ação
            reward, done, score = game.play_step(final_move)

            # Novo Estado
            state_new = agent.get_state(game)

            # Short Memory
            agent.train_short_memory(state_old,final_move,reward,state_new,done)

            # Replay memory
            agent.remember(state_old,final_move,reward,state_new,done)

            # Fim da partida
            if done:
                game.reset()
                agent.n_game += 1

                
                # Long memory
                agent.train_long_memory()

                # Log (sempre loga, recorde ou não)
                is_new_record = score > agent.record

                logger.log_game(score)

                # Atualiza recorde e salva checkpoint 
                # So quando ha melhora 
                if is_new_record:
                    agent.record = score
                    print(
                        f"Novo recorde: "
                        f"{agent.record}",
                        flush=True
                    )
                    agent.save_checkpoint()

    except KeyboardInterrupt:
        print("\nTreinamento interrompido.",flush=True)
        print("Salvando checkpoint de continuidade...",flush=True)

        agent.save_checkpoint()
        print("Checkpoint salvo.",flush=True)

if __name__ == "__main__":
    train()