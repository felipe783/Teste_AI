# Sobre o Arquivo

O arquivo `agent.py` é o **Orquestrador do Treino DQN v1**: monta o estado, escolhe ações, guarda experiências, chama o treino e salva checkpoints e vídeos.

Hiperparâmetros:

| Constante | Valor |
|---|---|
| `MAX_MEMORY` | 100 000 transições |
| `BATCH_SIZE` | 1 000 |
| `LR` | 0.001 |
| `gamma` | 0.9 |
| `LONG_TRAIN_INTERVAL` | a cada 10 partidas |
| `CHECKPOINT_INTERVAL` | a cada 100 partidas |
# Funcionalidades

## get_state (11 entradas)

````python
state = [
    perigo_reto, perigo_direita, perigo_esquerda,   # colisão na próxima célula
    dir_l, dir_r, dir_u, dir_d,                      # direção atual (one-hot)
    comida_esq, comida_dir, comida_cima, comida_baixo,
]
return np.array(state, dtype=np.int8)
````

## get_action (ε-greedy)

````python
self.epsilon = max(5, 80 - self.n_game)
if random.randint(0, 200) < self.epsilon:
    move = random.randint(0, 2)            # exploração
else:
    prediction = self.model(state0)
    move = torch.argmax(prediction).item() # aproveitamento
````

> [!warning] Epsilon do checkpoint é ignorado
> O `epsilon` carregado no `__init__` é sobrescrito na primeira ação. Depois de 75 partidas, a exploração fica fixa em `5/201 ≈ 2,5%`, seja num treino novo ou retomado.

## Loop de Treino

- **A cada passo**: `train_short_memory` (1 transição) e `remember`.
- **Ao fim da partida**: `reset`; a cada 10 partidas, `train_long_memory` com 1 000 amostras do buffer.
- **Vídeo**: salvo quando `score >= record - 5`, em `videos/` relativo ao diretório atual. Isso pode gerar muitos arquivos.
- **Checkpoint**: a cada 100 partidas e no `Ctrl+C`, via [[Rede e QTrainer|save_checkpoint]].
- **Gráfico**: `Helper.plot` redesenha score e média com Matplotlib/IPython a cada partida.

> [!note] Checkpoint versionado
> `models/checkpoint.pth` (no Git) tem `n_game = 11526`, `record = 120` e rede `11 → 256 → 3`. `models/model.pth` é um `state_dict` puro, do formato antigo. Inspecionado com [[Leitor PTH]] em 2026-10-17.
