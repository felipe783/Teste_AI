# Sobre o Arquivo

> [!warning] Código removido em 2026-10-29
> A pasta `SnakeGame/DQN/` (v1) foi apagada do repositório no commit `2e9f18e`. Esta nota fica como registro histórico, porque a [[Server DQN|v2]] é comparada com ela. Para ler o código antigo: `git show dfbc504:SnakeGame/DQN/<arquivo>`.

O arquivo `model.py` define a **Rede Neural** (`Linear_QNet`) e o **Treinador** (`QTrainer`), que aplica a equação de Bellman.
# Linear_QNet

Rede *feedforward* de duas camadas: `entrada → Linear → ReLU → Linear → 3 valores Q`.

````python
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)
````

## Checkpoint

- `save_checkpoint` grava em `DQN/models/checkpoint.pth` (caminho baseado em `__file__`): `model_state_dict`, `optimizer_state_dict`, `n_game`, `record`, `epsilon`.
- `load_checkpoint` restaura tudo, ou retorna `(0, 0, 80)` se o arquivo não existir.
# QTrainer

Responsabilidade:
- Calcular o alvo `Q_novo = r + γ · max Q(s')` (ou só `r` se terminou) e reduzir o erro com **Adam + MSELoss**.

````python
pred = self.model(state)
with torch.no_grad():
    max_next_q = torch.max(self.model(next_state), dim=1).values

target = pred.detach().clone()
q_new = reward.clone()
q_new[~done_tensor] += self.gamma * max_next_q[~done_tensor]
target[batch_indices, action_index] = q_new

loss = self.criterion(pred, target)
````

- Só o Q da ação tomada é alterado no `target`; as outras saídas ficam iguais à previsão, então têm erro zero.
- O mesmo `train_step` aceita 1 transição (`unsqueeze(0)`) ou um lote.

> [!warning] Sem Target Network
> `max Q(s')` vem da **mesma rede** que está sendo atualizada: o alvo se move a cada passo, o que tende a instabilizar o treino. A [[Server DQN|v2]] corrige isso com target network e Double DQN.
