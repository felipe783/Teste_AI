import os
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ============================================================
# CPU
# ============================================================

DEVICE = torch.device("cpu")

torch.set_num_threads(4)


# ============================================================
# LOG
# ============================================================

LOG_DIR = "logs"
LOG_FILE = os.path.join(
    LOG_DIR,
    "training_log.csv"
)


class TrainingLogger:

    def __init__(self):

        os.makedirs(
            LOG_DIR,
            exist_ok=True
        )

        self.total_games = 0
        self.record = 0

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
            f"Total: {self.total_games}"
        )


# ============================================================
# Q-NETWORK
# ============================================================

class Linear_QNet(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):

        super().__init__()

        self.linear1 = nn.Linear(
            input_size,
            hidden_size
        )

        self.linear2 = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        x = F.relu(
            self.linear1(x)
        )

        x = self.linear2(x)

        return x


# ============================================================
# TRAINER
# ============================================================

class QTrainer:

    def __init__(
        self,
        model,
        lr,
        gamma
    ):

        self.lr = lr
        self.gamma = gamma
        self.model = model

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr
        )

        self.criterion = nn.MSELoss()

    # ========================================================
    # TRAIN STEP
    # ========================================================

    def train_step(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        # ----------------------------------------------------
        # TENSORS
        # ----------------------------------------------------

        state = torch.as_tensor(
            state,
            dtype=torch.float32
        )

        next_state = torch.as_tensor(
            next_state,
            dtype=torch.float32
        )

        action = torch.as_tensor(
            action,
            dtype=torch.float32
        )

        reward = torch.as_tensor(
            reward,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # TRANSFORMA EM BATCH
        # ----------------------------------------------------

        if state.ndim == 1:

            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)

            done = (done,)

        else:

            done = tuple(done)

        # ----------------------------------------------------
        # Q ATUAL
        # ----------------------------------------------------

        pred = self.model(state)

        # ----------------------------------------------------
        # Q DO PRÓXIMO ESTADO
        # ----------------------------------------------------

        with torch.no_grad():

            next_pred = self.model(
                next_state
            )

            max_next_q = torch.max(
                next_pred,
                dim=1
            ).values

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        target = pred.detach().clone()

        done_tensor = torch.tensor(
            done,
            dtype=torch.bool
        )

        q_new = reward.clone()

        q_new[~done_tensor] += (
            self.gamma *
            max_next_q[~done_tensor]
        )

        # ----------------------------------------------------
        # AÇÃO ESCOLHIDA
        # ----------------------------------------------------

        action_index = torch.argmax(
            action,
            dim=1
        )

        batch_indices = torch.arange(
            target.size(0)
        )

        target[
            batch_indices,
            action_index
        ] = q_new

        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        self.optimizer.zero_grad(
            set_to_none=True
        )

        loss = self.criterion(
            pred,
            target
        )

        loss.backward()

        self.optimizer.step()