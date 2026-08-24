import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

DEVICE = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR,"models")
torch.set_num_threads(4)

class Linear_QNet(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save_checkpoint(self,optimizer,n_game,record,epsilon,file_name="checkpoint.pth"):

        os.makedirs(MODEL_DIR,exist_ok=True)
        file_path = os.path.join(MODEL_DIR,file_name)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "n_game": n_game,
            "record": record,
            "epsilon": epsilon
        }

        torch.save(checkpoint,file_path)
        print(f"Checkpoint salvo em: {file_path}")

    def load_checkpoint(self,optimizer,file_name="checkpoint.pth"):

        file_path = os.path.join(MODEL_DIR,file_name)

        if not os.path.exists(file_path):
            print(
                f"Nenhum checkpoint encontrado em: "
                f"{file_path}"
            )
            return 0, 0, 80

        checkpoint = torch.load(file_path,map_location="cpu",weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        n_game = checkpoint["n_game"]
        record = checkpoint["record"]
        epsilon = checkpoint["epsilon"]

        print("Checkpoint carregado!")
        print(f"Jogos anteriores: {n_game}")
        print(f"Recorde anterior: {record}")
        print(f"Epsilon anterior: {epsilon}")
        return n_game, record, epsilon

class QTrainer:

    def __init__(self,model,lr,gamma):

        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self,state,action,reward,next_state,done):

        state = torch.as_tensor(state,dtype=torch.float32)
        next_state = torch.as_tensor(next_state,dtype=torch.float32)
        action = torch.as_tensor(action,dtype=torch.float32)
        reward = torch.as_tensor(reward,dtype=torch.float32)

        if state.ndim == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)
        else:
            done = tuple(done)

        pred = self.model(state)

        with torch.no_grad():
            next_pred = self.model(next_state)
            max_next_q = torch.max(next_pred,dim=1).values

        target = pred.detach().clone()
        done_tensor = torch.tensor(done,dtype=torch.bool)
        q_new = reward.clone()
        q_new[~done_tensor] += (self.gamma *max_next_q[~done_tensor])

        action_index = torch.argmax(action,dim=1)
        batch_indices = torch.arange(target.size(0))
        target[
            batch_indices,
            action_index
        ] = q_new

        self.optimizer.zero_grad(set_to_none=True)

        loss = self.criterion(pred,target)
        loss.backward()
        self.optimizer.step()