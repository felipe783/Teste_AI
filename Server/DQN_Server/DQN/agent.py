import os
import csv
import torch
import random
import numpy as np

from collections import deque

from snake_gameai import (
    SnakeGameAI,
    Direction,
    Point,
    BLOCK_SIZE
)

from model import (
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

MODEL_DIR = "Models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model.pth"
)

LOG_DIR = "logs"
LOG_FILE = os.path.join(
    LOG_DIR,
    "training_log.csv"
)


# ============================================================
# LOGGER
# ============================================================

class TrainingLogger:

    def __init__(self):

        self.total_games = 0
        self.record = 0

        os.makedirs(
            LOG_DIR,
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

        file_exists = os.path.exists(
            LOG_FILE
        )

        with open(
            LOG_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

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


# ============================================================
# AGENT
# ============================================================

class Agent:

    def __init__(self):

        self.n_game = 0
        self.record = 0

        # ----------------------------------------------------
        # Exploração
        # ----------------------------------------------------

        self.epsilon = 80

        # ----------------------------------------------------
        # Desconto futuro
        # ----------------------------------------------------

        self.gamma = 0.9

        # ----------------------------------------------------
        # Replay memory
        # ----------------------------------------------------

        self.memory = deque(
            maxlen=MAX_MEMORY
        )

        # ----------------------------------------------------
        # Modelo
        # ----------------------------------------------------

        self.model = Linear_QNet(
            11,
            256,
            3
        )

        # ----------------------------------------------------
        # Trainer
        # ----------------------------------------------------

        self.trainer = QTrainer(
            self.model,
            lr=LR,
            gamma=self.gamma
        )

        # ----------------------------------------------------
        # Carrega checkpoint
        # ----------------------------------------------------

        self.load_checkpoint()

    # ========================================================
    # CHECKPOINT - CARREGAR
    # ========================================================

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

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        # ----------------------------------------------------
        # Compatibilidade com checkpoint completo
        # ----------------------------------------------------

        if isinstance(
            checkpoint,
            dict
        ) and "model_state_dict" in checkpoint:

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            if "optimizer_state_dict" in checkpoint:

                self.trainer.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"]
                )

            self.n_game = checkpoint.get(
                "n_game",
                0
            )

            self.record = checkpoint.get(
                "record",
                0
            )

        # ----------------------------------------------------
        # Compatibilidade com state_dict antigo
        # ----------------------------------------------------

        else:

            self.model.load_state_dict(
                checkpoint
            )

        self.model.to(DEVICE)

        print(
            f"Checkpoint carregado!",
            flush=True
        )

        print(
            f"Partida inicial: {self.n_game}",
            flush=True
        )

        print(
            f"Recorde inicial: {self.record}",
            flush=True
        )

    # ========================================================
    # CHECKPOINT - SALVAR
    # ========================================================

    def save_checkpoint(self):

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        checkpoint = {

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.trainer.optimizer.state_dict(),

            "n_game":
                self.n_game,

            "record":
                self.record
        }

        torch.save(
            checkpoint,
            MODEL_PATH
        )

        print(
            f"Checkpoint salvo em: "
            f"{MODEL_PATH}",
            flush=True
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

            # ------------------------------------------------
            # Perigo à frente
            # ------------------------------------------------

            (
                (dir_u and game.is_collision(point_u)) or
                (dir_d and game.is_collision(point_d)) or
                (dir_l and game.is_collision(point_l)) or
                (dir_r and game.is_collision(point_r))
            ),

            # ------------------------------------------------
            # Perigo à direita
            # ------------------------------------------------

            (
                (dir_u and game.is_collision(point_r)) or
                (dir_d and game.is_collision(point_l)) or
                (dir_l and game.is_collision(point_u)) or
                (dir_r and game.is_collision(point_d))
            ),

            # ------------------------------------------------
            # Perigo à esquerda
            # ------------------------------------------------

            (
                (dir_u and game.is_collision(point_l)) or
                (dir_d and game.is_collision(point_r)) or
                (dir_l and game.is_collision(point_d)) or
                (dir_r and game.is_collision(point_u))
            ),

            # ------------------------------------------------
            # Direção
            # ------------------------------------------------

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # ------------------------------------------------
            # Comida
            # ------------------------------------------------

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

        final_move = [
            0,
            0,
            0
        ]

        # ----------------------------------------------------
        # Exploração
        # ----------------------------------------------------

        if random.randint(
            0,
            200
        ) < self.epsilon:

            move = random.randint(
                0,
                2
            )

        # ----------------------------------------------------
        # Modelo
        # ----------------------------------------------------

        else:

            state0 = torch.as_tensor(
                np.array(state),
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

    print(
        "==============================================",
        flush=True
    )

    print(
        "       TREINAMENTO DQN INICIADO",
        flush=True
    )

    print(
        f"Modelo: {MODEL_PATH}",
        flush=True
    )

    print(
        f"Partida inicial: {agent.n_game}",
        flush=True
    )

    print(
        f"Recorde inicial: {agent.record}",
        flush=True
    )

    print(
        "==============================================",
        flush=True
    )

    try:

        while True:

            # ------------------------------------------------
            # Estado atual
            # ------------------------------------------------

            state_old = agent.get_state(
                game
            )

            # ------------------------------------------------
            # Escolhe ação
            # ------------------------------------------------

            final_move = agent.get_action(
                state_old
            )

            # ------------------------------------------------
            # Executa ação
            # ------------------------------------------------

            reward, done, score = game.play_step(
                final_move
            )

            # ------------------------------------------------
            # Novo estado
            # ------------------------------------------------

            state_new = agent.get_state(
                game
            )

            # ------------------------------------------------
            # Short memory
            # ------------------------------------------------

            agent.train_short_memory(
                state_old,
                final_move,
                reward,
                state_new,
                done
            )

            # ------------------------------------------------
            # Replay memory
            # ------------------------------------------------

            agent.remember(
                state_old,
                final_move,
                reward,
                state_new,
                done
            )

            # ------------------------------------------------
            # Fim da partida
            # ------------------------------------------------

            if done:

                game.reset()

                agent.n_game += 1

                # ------------------------------------------------
                # Long memory
                # ------------------------------------------------

                agent.train_long_memory()

                # ------------------------------------------------
                # Log
                # ------------------------------------------------

                logger.log_game(
                    score
                )

                # ------------------------------------------------
                # Atualiza recorde
                # ------------------------------------------------

                if score > agent.record:

                    agent.record = score

                    print(
                        f"Novo recorde: "
                        f"{agent.record}",
                        flush=True
                    )

                # ------------------------------------------------
                # Salva checkpoint
                # ------------------------------------------------

                agent.save_checkpoint()

    except KeyboardInterrupt:

        print(
            "\nTreinamento interrompido.",
            flush=True
        )

        print(
            "Salvando checkpoint...",
            flush=True
        )

        agent.save_checkpoint()

        print(
            "Checkpoint salvo.",
            flush=True
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()