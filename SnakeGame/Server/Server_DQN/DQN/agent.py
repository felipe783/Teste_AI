import argparse
import csv
import os
import random
from collections import deque
from statistics import mean, median
import numpy as np
import torch
from model import DEVICE, Linear_QNet, QTrainer
from snake_gameai import BLOCK_SIZE, Direction, SnakeGameAI

MAX_MEMORY = 100_000
BATCH_SIZE = 256
LR = 2.5e-4
STATE_SIZE = 21
HIDDEN_SIZE = 256
GAMMA = 0.95
TARGET_UPDATE_FREQ = 1_000
EPSILON_START = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY_STEPS = 250_000
CHECKPOINT_EVERY_GAMES = 100

MODEL_DIR = "Models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")
LOG_MAX_SIZE_BYTES = 5 * 1024 * 1024 * 1024


class TrainingLogger:
    def __init__(self):
        self.total_games = self.record = 0
        os.makedirs(LOG_DIR, exist_ok=True)

        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    try:
                        self.total_games = max(self.total_games, int(row["partida"]))
                        self.record = max(self.record, int(row["max"]))
                    except (ValueError, KeyError):
                        pass

    def sync_with_agent(self, agent):
        self.total_games = max(self.total_games, agent.n_game)
        self.record = max(self.record, agent.record)

    def log_game(self, score, epsilon):
        self.total_games += 1
        self.record = max(self.record, score)

        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) >= LOG_MAX_SIZE_BYTES:
            os.remove(LOG_FILE)
        exists = os.path.exists(LOG_FILE)

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow(["partida", "score", "max", "epsilon"])
            writer.writerow([self.total_games, score, self.record, f"{epsilon:.5f}"])

        print(f"Partida: {self.total_games} | Score: {score} | Max: {self.record} | " f"Epsilon: {epsilon:.4f}", flush=True)

class Agent:
    def __init__(self, load_checkpoint=True):
        self.n_game = self.record = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(STATE_SIZE, HIDDEN_SIZE, 3).to(DEVICE)
        self.trainer = QTrainer(self.model, LR, GAMMA, TARGET_UPDATE_FREQ)
        self.checkpoint_loaded = False
        if load_checkpoint:
            self.load_checkpoint()

    @property
    def epsilon(self):
        fraction = min(1.0, self.trainer.train_steps / EPSILON_DECAY_STEPS)
        return EPSILON_START + fraction * (EPSILON_MIN - EPSILON_START)

    def load_checkpoint(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Nenhum checkpoint encontrado em: {MODEL_PATH}", flush=True)
            return False
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Não foi possível ler o checkpoint; iniciando rede nova: {error}", flush=True)
            return False

        # Aceita tanto o formato novo quanto um state_dict puro de versões antigas.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "linear1.weight" in checkpoint:
            state_dict = checkpoint
        else:
            state_dict = None

        if not isinstance(state_dict, dict) or "linear1.weight" not in state_dict:
            print("Checkpoint sem pesos válidos; iniciando rede nova. "
                  "O arquivo será substituído no próximo salvamento.", flush=True)
            return False

        saved_size = state_dict["linear1.weight"].shape[1]
        if saved_size != STATE_SIZE:
            print(f"Checkpoint incompatível: estado={saved_size}, esperado={STATE_SIZE}. "
                  "Iniciando uma rede nova para não misturar representações.", flush=True)
            return False
        self.model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict):
            if "optimizer_state_dict" in checkpoint:
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "target_model_state_dict" in checkpoint:
                self.trainer.load_target_state_dict(checkpoint["target_model_state_dict"])
            else:
                self.trainer.update_target_model()
            self.n_game = checkpoint.get("n_game", 0)
            self.record = checkpoint.get("record", 0)

            # Formato antigo não possuía train_steps; assim volta a explorar.
            self.trainer.train_steps = checkpoint.get("train_steps", 0)
        else:
            self.trainer.update_target_model()
        self.checkpoint_loaded = True
        print(f"Checkpoint carregado: partidas={self.n_game}, recorde={self.record}, "
              f"train_steps={self.trainer.train_steps}", flush=True)
        return True

    def save_checkpoint(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save({"checkpoint_version": 2, "state_size": STATE_SIZE,
                    "model_state_dict": self.model.state_dict(),
                    "target_model_state_dict": self.trainer.target_model.state_dict(),
                    "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                    "n_game": self.n_game, "record": self.record,
                    "train_steps": self.trainer.train_steps}, MODEL_PATH)
        print(f"Checkpoint salvo: {MODEL_PATH}", flush=True)

    def get_state(self, game):
        head, tail = game.head, game.snake[-1]
        actions = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        analyses = [game.action_analysis(action) for action in actions]
        board_width, board_height = max(1, game.w - BLOCK_SIZE), max(1, game.h - BLOCK_SIZE)
        capacity = (game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE)
        state = [
            *(game.direction == direction for direction in
              (Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN)),
            (game.food.x - head.x) / board_width, (game.food.y - head.y) / board_height,
            (tail.x - head.x) / board_width, (tail.y - head.y) / board_height,
            *(item[0] for item in analyses),       # colisão: frente, direita, esquerda
            *(item[1] for item in analyses),       # fração de espaço acessível
            *(item[2] for item in analyses),       # rota segura até a comida
            *(item[3] for item in analyses),       # rota até a cauda/ciclo de escape
            len(game.snake) / capacity,
        ]
        return np.asarray(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) >= BATCH_SIZE:
            self.trainer.train_step(*zip(*random.sample(self.memory, BATCH_SIZE)))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, explore=True):
        move = random.randrange(3) if explore and random.random() < self.epsilon else None
        if move is None:
            with torch.no_grad():
                move = self.model(torch.as_tensor(state, dtype=torch.float32, device=DEVICE)).argmax().item()
        action = [0, 0, 0]
        action[move] = 1
        return action

def train():
    agent, logger, game = Agent(), TrainingLogger(), SnakeGameAI()
    logger.sync_with_agent(agent)
    print(f"Treinamento DQN iniciado em CPU | modelo: {MODEL_PATH}", flush=True)

    try:
        while True:
            state_old = agent.get_state(game)
            action = agent.get_action(state_old)
            reward, done, score = game.play_step(action)
            state_new = agent.get_state(game)
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)
            if done:
                game.reset()
                agent.n_game += 1
                agent.train_long_memory()
                new_record = score > agent.record
                if new_record:
                    agent.record = score
                    print(f"Novo recorde: {score}", flush=True)
                logger.log_game(score, agent.epsilon)
                if new_record or agent.n_game % CHECKPOINT_EVERY_GAMES == 0:
                    agent.save_checkpoint()
    except KeyboardInterrupt:
        print("\nTreinamento interrompido; salvando continuidade...", flush=True)
        agent.save_checkpoint()


def evaluate(episodes):
    agent = Agent()
    if not agent.checkpoint_loaded:
        raise RuntimeError("Não há checkpoint compatível para avaliar.")
    agent.model.eval()
    scores = []
    for _ in range(episodes):
        game, done = SnakeGameAI(), False
        while not done:
            _, done, score = game.play_step(agent.get_action(agent.get_state(game), explore=False))
        scores.append(score)
    print(f"Avaliação ({episodes} partidas, epsilon=0): melhor={max(scores)}, "
          f"média={mean(scores):.2f}, mediana={median(scores):.2f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino e avaliação do DQN Snake")
    parser.add_argument("--eval", action="store_true", help="avalia sem treinar ou salvar")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    
    if args.eval:
        if args.episodes < 1:
            parser.error("--episodes deve ser pelo menos 1")
        evaluate(args.episodes)
    else:
        train()
