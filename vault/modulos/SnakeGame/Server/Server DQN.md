# Sobre o Arquivo

Os arquivos de `Server/Server_DQN/DQN/` formam o **DQN v2**: reescrita do [[Agente DQN|v1]] com estado mais rico, *target network*, Double DQN, *reward shaping* e log em CSV. É o modelo usado no [[Play]].

| Constante | v1 | v2 |
|---|---|---|
| Entradas | 11 | **21** |
| `BATCH_SIZE` | 1 000 | 256 |
| `LR` | 0.001 | 2.5e-4 |
| `GAMMA` | 0.9 | 0.95 |
| Loss | MSE | **SmoothL1 (Huber)** |
| ε | `max(5, 80 − partidas)/201` | linear de **1.0 → 0.02** em 250 000 passos de treino |
| Target network | — | atualizada a cada **1 000** passos |
| Treino longo | a cada 10 partidas | ao fim de **toda** partida |
# Funcionalidades

## get_state (21 entradas)

````python
state = [
    *direção_one_hot,                       # 4
    dx_comida, dy_comida,                   # 2 (normalizados)
    dx_cauda, dy_cauda,                     # 2
    *colisão[reto, dir, esq],               # 3
    *espaço_alcançável[reto, dir, esq],     # 3 (BFS)
    *comida_alcançável[reto, dir, esq],     # 3
    *cauda_alcançável[reto, dir, esq],      # 3
    len(snake) / capacidade,                # 1
]
````

As features vêm de `action_analysis` ([[Ambiente Snake]]), que simula cada ação e mede por BFS se a cobra ficaria presa.

## epsilon como propriedade

````python
@property
def epsilon(self):
    fraction = min(1.0, self.trainer.train_steps / EPSILON_DECAY_STEPS)
    return EPSILON_START + fraction * (EPSILON_MIN - EPSILON_START)
````

O decaimento depende de `train_steps`, que é salvo no checkpoint. Por isso, ao retomar, a exploração continua de onde parou.

## QTrainer com Double DQN

````python
current_q = self.model(state).gather(1, action_index).squeeze(1)
with torch.no_grad():
    next_actions = self.model(next_state).argmax(dim=1, keepdim=True)        # policy escolhe
    next_q = self.target_model(next_state).gather(1, next_actions).squeeze(1) # target avalia
    target_q = reward + self.gamma * next_q * (~done_tensor).float()
loss = self.criterion(current_q, target_q)
loss.backward()
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
````

- **Target network**: um alvo estável durante 1 000 passos reduz a perseguição de um "alvo móvel".
- **Double DQN**: separar quem escolhe a ação de quem a avalia reduz a superestimação de Q.
- **Huber + gradient clipping**: limitam o efeito de erros grandes.

## Checkpoint e Log

- `Models/model.pth` (relativo ao diretório atual): pesos, *target*, otimizador, `n_game`, `record`, `train_steps`, `checkpoint_version = 2`, `state_size`. É salvo a cada 100 partidas, a cada novo recorde e no `Ctrl+C`.
- `load_checkpoint` aceita o formato antigo e **recusa** checkpoints com número de entradas diferente, iniciando uma rede nova em vez de misturar representações.
- `TrainingLogger` grava `logs/training_log.csv` (`partida, score, max, epsilon`) e **apaga** o arquivo ao passar de 5 GB.

## Avaliação

````bash
python DQN/agent.py --eval --episodes 20
````

Roda sem exploração (`explore=False`) e imprime melhor, média e mediana.

> [!note] Modelo treinado
> `Test/Models/model.pth` (não versionado) tem `n_game = 14500`, `record = 101` e `train_steps = 7 591 701`; `model_antigo.pth` tem 6 400 partidas e recorde 90. Inspecionados em 2026-10-17.
