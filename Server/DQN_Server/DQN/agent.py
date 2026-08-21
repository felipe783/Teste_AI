import torch
import random
import numpy as np

from collections import deque

from Teste_IA.SnakeGameAI.DQN.snake_gameai import (
    SnakeGameAI,
    Direction,
    Point,
    BLOCK_SIZE
)

from Teste_IA.SnakeGameAI.Server.DQN_Server.DQN.model import (
    Linear_QNet,
    QTrainer,
    DEVICE
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


# ============================================================
# LOG
# ============================================================

LOG_FILE = "logs/training_log.csv"


class TrainingLogger:

    def __init__(self):

        import os
        import csv

        self.csv = csv
        self.os = os

        self.total_games = 0
        self.record = 0

        os.makedirs(
            "logs",
            exist_ok=True
        )

        if os.path.exists(LOG_FILE):

            with open(
                LOG_FILE,
                "r",
                newline="",
                encoding="utf-8"
            ) as file:

                reader = csv.DictReader(file)

                for row in reader:

                    self.total_games = int(
                        row["partida"]
                    )

                    self.record = max(
                        self.record,
                        int(row["max"])
                    )

    def log_game(self, score):

        self.total_games += 1

        self.record = max(
            self.record,
            score
        )

        file_exists = self.os.path.exists(
            LOG_FILE
        )

        with open(
            LOG_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = self.csv.writer(file)

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
            f"Total: {self.total_games}"
        )


# ============================================================
# AGENT
# ============================================================

class Agent:

    def __init__(self):

        self.n_game = 0

        self.record = 0

        # Exploração
        self.epsilon = 80

        # Desconto futuro
        self.gamma = 0.9

        # Replay memory
        self.memory = deque(
            maxlen=MAX_MEMORY
        )

        # Modelo
        self.model = Linear_QNet(
            11,
            256,
            3
        )

        # Trainer
        self.trainer = QTrainer(
            self.model,
            lr=LR,
            gamma=self.gamma
        )

    # ========================================================
    # ESTADO
    # ========================================================

    def get_state(self, game):

        head = game.snake[0]

        point_l = Point(
            head.x - BLOCK_SIZE,
            head.y
        )

        point_r = Point(
            head.x + BLOCK_SIZE,
            head.y
        )

        point_u = Point(
            head.x,
            head.y - BLOCK_SIZE
        )

        point_d = Point(
            head.x,
            head.y + BLOCK_SIZE
        )

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [

            # Perigo à frente
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)),

            # Perigo à direita
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Perigo à esquerda
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)),

            # Direção
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Comida
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y
        ]

        return np.array(
            state,
            dtype=np.int8
        )

    # ========================================================
    # MEMORY
    # ========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    # ========================================================
    # LONG MEMORY
    # ========================================================

    def train_long_memory(self):

        if len(self.memory) < BATCH_SIZE:
            return

        mini_sample = random.sample(
            self.memory,
            BATCH_SIZE
        )

        states, actions, rewards, next_states, dones = zip(
            *mini_sample
        )

        self.trainer.train_step(
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    # ========================================================
    # SHORT MEMORY
    # ========================================================

    def train_short_memory(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.trainer.train_step(
            state,
            action,
            reward,
            next_state,
            done
        )

    # ========================================================
    # ACTION
    # ========================================================

    def get_action(self, state):

        self.epsilon = max(
            5,
            80 - self.n_game
        )

        final_move = [0, 0, 0]

        # Exploração
        if random.randint(0, 200) < self.epsilon:

            move = random.randint(0, 2)

        # Exploração pelo modelo
        else:

            state0 = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=DEVICE
            )

            with torch.no_grad():

                prediction = self.model(
                    state0
                )

            move = torch.argmax(
                prediction
            ).item()

        final_move[move] = 1

        return final_move


# ============================================================
# TREINAMENTO
# ============================================================

def train():

    agent = Agent()

    logger = TrainingLogger()

    game = SnakeGameAI()

    try:

        while True:

            # ------------------------------------------------
            # ESTADO ATUAL
            # ------------------------------------------------

            state_old = agent.get_state(
                game
            )

            # ------------------------------------------------
            # ESCOLHE AÇÃO
            # ------------------------------------------------

            final_move = agent.get_action(
                state_old
            )

            # ------------------------------------------------
            # EXECUTA AÇÃO
            # ------------------------------------------------

            reward, done, score = game.play_step(
                final_move
            )

            # ------------------------------------------------
            # NOVO ESTADO
            # ------------------------------------------------

            state_new = agent.get_state(
                game
            )

            # ------------------------------------------------
            # SHORT MEMORY
            # ------------------------------------------------

            agent.train_short_memory(
                state_old,
                final_move,
                reward,
                state_new,
                done
            )

            # ------------------------------------------------
            # REPLAY MEMORY
            # ------------------------------------------------

            agent.remember(
                state_old,
                final_move,
                reward,
                state_new,
                done
            )

            # ------------------------------------------------
            # FIM DA PARTIDA
            # ------------------------------------------------

            if done:

                game.reset()

                agent.n_game += 1

                # ------------------------------------------------
                # LONG MEMORY
                # ------------------------------------------------

                agent.train_long_memory()

                # ------------------------------------------------
                # LOG
                # ------------------------------------------------

                logger.log_game(
                    score
                )

    except KeyboardInterrupt:

        print(
            "\nTreinamento interrompido."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()