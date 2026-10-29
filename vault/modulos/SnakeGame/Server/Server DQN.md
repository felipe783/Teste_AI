# Sobre o Arquivo

Os arquivos de `Server/Server_DQN/DQN/` formam o **DQN v2**: reescrita do [[Agente DQN|v1]] com estado mais rico, *target network*, Double DQN, *reward shaping* e log em CSV. É o modelo usado no [[Play]].

| Constante      | v1                                                  | v2                                                                                                           |
| -------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Entradas       | 11                                                  | **21**                                                                                                       |
| `BATCH_SIZE`   | 1 000                                               | 256                                                                                                          |
| `LR`           | 0.001                                               | 2.5e-4                                                                                                       |
| `GAMMA`        | 0.9                                                 | 0.95                                                                                                         |
| Loss           | MSE                                                 | **SmoothL1 (Huber)**                                                                                         |
| ε              | `max(5, 80 − partidas)/201`                         | linear de **1.0 → 0.02** em 250 000 passos do ambiente (62 500 passos de treino)                             |
| Target network | —                                                   | atualizada a cada **1 000** passos do ambiente (250 passos de treino)                                        |
| Treino         | lote 1 a cada passo + lote 1 000 a cada 10 partidas | **lote 256 do replay a cada 4 passos** (desde 2026-10-17; antes: lote 1 a cada passo + lote 256 por partida) |
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

- `Models/model.pth` (relativo ao diretório atual): **último** estado — pesos, *target*, otimizador, `n_game`, `record`, `train_steps`, `best_eval_mean`, `train_every_steps`, `checkpoint_version = 2`, `state_size`. É salvo a cada 100 partidas, a cada novo recorde e no `Ctrl+C`.
- `Models/best_model.pth`: **melhor** modelo pela avaliação gulosa (ver [[#Regressão 15k × 30k (2026-10-17)]]). Só é sobrescrito quando a média melhora; `best_eval_mean` vai junto no `model.pth` para que uma retomada não rebaixe o melhor.
- `logs/eval_log.csv`: `partida, media, mediana, min, max, melhor_media, loss_media` a cada avaliação.
- `load_checkpoint` aceita o formato antigo e **recusa** checkpoints com número de entradas diferente, iniciando uma rede nova em vez de misturar representações.
- `TrainingLogger` grava `logs/training_log.csv` (`partida, score, max, epsilon`) e **apaga** o arquivo ao passar de 5 GB.

## Avaliação

````bash
python DQN/agent.py --eval --episodes 20
python DQN/agent.py --eval --episodes 50 --model Models/m15k.pth Models/m30k.pth   # comparação pareada
python DQN/agent.py --games 15000 --seed 0                                           # treino com orçamento e semente
````

Roda sem exploração (`explore=False`). A partida *i* usa `random.seed(seed + i)` (padrão `EVAL_SEED = 12345`), então todos os modelos enfrentam a mesma sequência de comidas e, como a política gulosa é determinística, o resultado de um checkpoint é reprodutível. Imprime média ± IC95%, mediana, mínimo e máximo; com dois modelos, imprime a **diferença pareada** por partida e se ela é significativa.

> [!note] Modelo treinado
> `Test/Models/model.pth` (não versionado) tem `n_game = 14500`, `record = 101` e `train_steps = 7 591 701`; `model_antigo.pth` tem 6 400 partidas e recorde 90. Inspecionados em 2026-10-17.

# Regressão 15k × 30k (2026-10-17)

**Problema relatado:** o modelo com ~15 000 partidas joga melhor que o de ~30 000. O modelo de 30 000 **não estava disponível localmente**, então essa comparação não foi refeita.

## Lido no código

- A descrição "lote de 1 000 ao fim de cada partida" **não bate com nenhuma versão**: a v1 usa lote de 1 000 a cada 10 partidas, e a v2 usava lote de 256 a cada partida.
- Na v2, quase todo o aprendizado era **online**: `train_short_memory` dava 1 passo de gradiente com lote 1 a **cada passo** do ambiente. Pelo checkpoint de 14 500 partidas (7,59 M `train_steps`), são ~523 passos online para 1 do replay por partida. Como o Adam normaliza o tamanho de cada passo, um lote de 1 move os pesos tanto quanto um de 256: o replay quase não pesava, e a rede seguia as transições recentes, que são correlacionadas.
- `model.pth` era sobrescrito a cada 100 partidas e a cada recorde, **sem avaliação**. "15k" e "30k" eram fotos de um processo oscilante, não o melhor modelo de cada fase.
- Já existiam e foram mantidos: target network, Double DQN, Huber, clipping de gradiente e terminal tratado corretamente (`~done`).

## Executado

| Experimento | Resultado |
|---|---|
| Modelos existentes, 20 partidas gulosas pareadas (seed 12345) | 6 400 partidas: **2,1 ± 1,1** (apesar do recorde 90) · 14 500: **102,0 ± 16,3** · diferença +99,9 ± 16,0 |
| A/B do zero, 1 000 partidas, avaliação final com seed 777 (nunca usada na seleção) | ver tabela abaixo |

| Semente | A: original | B: lote 256 a cada 4 passos | B − A (pareado) |
|---|---|---|---|
| 0 | 58,1 ± 12,4 (mediana 64) | 83,5 ± 12,3 (mediana 88) | **+25,5 ± 14,3** |
| 1 | 13,1 ± 9,2 (mediana **0**) | 79,6 ± 13,7 (mediana 84) | **+66,5 ± 14,9** |

Condições do A/B: o mesmo ambiente (otimizado; hash idêntico ao original), a mesma rede e as mesmas recompensas, ε decaindo em 50 000 passos do ambiente nas duas variantes (no lugar de 250 000, para caber em ~1 h) e 1 thread por processo. As cópias e os modelos do experimento ficaram fora do repositório.

## Conclusões

- **Confirmado:** com o mesmo orçamento, a nova regra de atualização aprendeu mais e de forma mais consistente entre sementes (A variou de 13 a 58; B, de 80 a 84). A mediana 0 de A na semente 1 mostra uma política gulosa que, na maioria das partidas, nem chega à primeira comida.
- **Confirmado:** o recorde de uma partida não indica a qualidade do checkpoint (6 400 partidas: recorde 90, média gulosa 2,1).
- **Provável, não verificado:** a regressão 15k → 30k é a mesma oscilação vista em escala longa. Só um treino longo com o código novo confirma; o `best_model.pth` protege o melhor modelo nesse meio-tempo.
- **Descartado por ora:** PER. O replay quase não era usado, então priorizar a amostragem dele não atacava a causa. Pode ser reavaliado se a curva de `eval_log.csv` estagnar.

## Limitações e próximos passos

1. O A/B cobre só 1 000 partidas e 2 sementes, com ε encurtado. Ele não prova nada sobre o regime de 15k–30k.
2. Treinar no servidor do zero com o código novo (fazer backup de `Models/` e `logs/` antes) e acompanhar `logs/eval_log.csv` até 30 000 partidas.
3. Comparar `best_model.pth` com o modelo antigo de 30k: `python DQN/agent.py --eval --episodes 50 --seed 777 --model <antigo> Models/best_model.pth`.
4. Não alterados: a v1 (`SnakeGame/DQN`) tem o mesmo treino online; a cópia de `snake_gameai.py` em `Test/` não recebeu a otimização do BFS.
