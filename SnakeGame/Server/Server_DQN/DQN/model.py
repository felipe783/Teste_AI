import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

"""Este arquivo cria a Rede Neural e compara o que a rede previu com o que realmente aconteceu (recompensa + estimativa do próximo estado) e ajusta os pesos para reduzir esse erro."""

class Linear_QNet(nn.Module): # Rede Neural
    # Rede Feedforward de duas camadas
    # Uma oculta e uma de saida
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))
    """O Relu quebra a linearidade e possibilita e rede aprender comportamentos mais complexos"""


class QTrainer:
    """DQN com target network fixa e bootstrap Double DQN."""

    def __init__(self, model, lr, gamma, target_update_freq=1000):
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(DEVICE) # Policy Network
        self.target_model = copy.deepcopy(model).to(DEVICE) # Target Network
        self.target_model.eval()

        for parameter in self.target_model.parameters():
            parameter.requires_grad = False # garante que ela nunca receba gradiente

        self.target_update_freq = target_update_freq
        self.train_steps = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss() # Huber Loss

    """ Copia de pesos da PN para a TN """
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def load_target_state_dict(self, state_dict):
        self.target_model.load_state_dict(state_dict)
        self.target_model.eval()

    def train_step(self, state, action, reward, next_state, done):
        # Tensor
        state = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=DEVICE)
        action = torch.as_tensor(action, dtype=torch.float32, device=DEVICE)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=DEVICE)

        if state.ndim == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        done_tensor = torch.as_tensor(done, dtype=torch.bool, device=DEVICE)
        action_index = action.argmax(dim=1, keepdim=True)
        current_q = self.model(state).gather(1, action_index).squeeze(1)

        # Double DQN: a policy escolhe a ação; a target a avalia.
        with torch.no_grad():
            next_actions = self.model(next_state).argmax(dim=1, keepdim=True) # Melhor acao tomada pela PN
            next_q = self.target_model(next_state).gather(1, next_actions).squeeze(1) # Melhor acao segundo a TN
            target_q = reward + self.gamma * next_q * (~done_tensor).float() 
            """ (~done_tensor).float()  evita que a Rede calcule uma recompensa futura depois de ter morrido """

        # Loss, backprop e clipping
        self.optimizer.zero_grad(set_to_none=True) # Limpa os gradientes anteriores

        """ O Loss mede o quanto a previsão da IA está diferente do valor que ela deveria prever. """
        loss = self.criterion(current_q, target_q) 
        loss.backward() # Backpropagation

        """ Gradient Clipping """
        # Limita o tamanho dos gradientes 
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.train_steps += 1

        # Update da TN
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_model() 
        return loss.item()
