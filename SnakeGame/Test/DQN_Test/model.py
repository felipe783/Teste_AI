import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


DEVICE = torch.device("cpu")

torch.set_num_threads(4)

class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class QTrainer:

    def __init__(self, model, lr, gamma, target_update_freq=1000):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = Linear_QNet(
            model.linear1.in_features,
            model.linear1.out_features,
            model.linear2.out_features
        )

        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

        for param in self.target_model.parameters():
            param.requires_grad = False

        self.target_update_freq = target_update_freq
        self.train_steps = 0
        self.optimizer = optim.Adam(model.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load_target_state_dict(self, state_dict):
        self.target_model.load_state_dict(state_dict)
        self.target_model.eval()

        for param in self.target_model.parameters():
            param.requires_grad = False

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

        # Q atual
        pred = self.model(state)

        # Q próximo estado
        with torch.no_grad():
            next_pred = self.target_model(next_state)
            max_next_q = torch.max(next_pred,dim=1).values
        # Target
        target = pred.detach().clone()
        done_tensor = torch.tensor(done,dtype=torch.bool)

        q_new = reward.clone()
        q_new[~done_tensor] += (
            self.gamma *
            max_next_q[~done_tensor]
        )

        # Ação escolhida
        action_index = torch.argmax(action,dim=1)
        batch_indices = torch.arange(target.size(0))

        target[
            batch_indices,
            action_index
        ] = q_new

        # Backpropagation
        self.optimizer.zero_grad(set_to_none=True)

        loss = self.criterion(pred,target)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1.0)

        self.optimizer.step()

        # Atualização Target Network
        self.train_steps += 1

        if self.train_steps % self.target_update_freq == 0:
            self.update_target_model()