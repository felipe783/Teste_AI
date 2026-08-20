import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Pasta onde está o model.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pasta dos modelos
MODEL_DIR = os.path.join(BASE_DIR, "models")


class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

    def save_checkpoint(
        self,
        optimizer,
        n_game,
        record,
        epsilon,
        file_name="checkpoint.pth"
    ):

        # Cria a pasta models caso não exista
        os.makedirs(MODEL_DIR, exist_ok=True)

        file_path = os.path.join(MODEL_DIR, file_name)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "n_game": n_game,
            "record": record,
            "epsilon": epsilon
        }

        torch.save(checkpoint, file_path)

        print(f"Checkpoint salvo em: {file_path}")

    def load_checkpoint(
        self,
        optimizer,
        file_name="checkpoint.pth"
    ):

        file_path = os.path.join(MODEL_DIR, file_name)

        # Se não existir checkpoint
        if not os.path.exists(file_path):
            print(f"Nenhum checkpoint encontrado em: {file_path}")

            return 0, 0, 80

        # Carrega checkpoint
        checkpoint = torch.load(
            file_path,
            weights_only=False
        )

        # Carrega pesos da rede
        self.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Carrega estado do otimizador
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        # Recupera informações do treinamento
        n_game = checkpoint["n_game"]
        record = checkpoint["record"]
        epsilon = checkpoint["epsilon"]

        print("Checkpoint carregado!")
        print(f"Jogos anteriores: {n_game}")
        print(f"Recorde anterior: {record}")
        print(f"Epsilon anterior: {epsilon}")

        return n_game, record, epsilon


class QTrainer:

    def __init__(self, model, lr, gamma):

        self.lr = lr
        self.gamma = gamma
        self.model = model

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr
        )

        self.criterion = nn.MSELoss()

    def train_step(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        state = torch.tensor(
            state,
            dtype=torch.float
        )

        next_state = torch.tensor(
            next_state,
            dtype=torch.float
        )

        action = torch.tensor(
            action,
            dtype=torch.long
        )

        reward = torch.tensor(
            reward,
            dtype=torch.float
        )

        if len(state.shape) == 1:

            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)

            done = (done,)

        # Predição atual
        pred = self.model(state)

        # Cópia da predição
        target = pred.clone()

        for idx in range(len(done)):

            Q_new = reward[idx]

            if not done[idx]:

                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )

            target[idx][
                torch.argmax(action[idx]).item()
            ] = Q_new

        # Backpropagation
        self.optimizer.zero_grad()

        loss = self.criterion(
            target,
            pred
        )

        loss.backward()

        self.optimizer.step()