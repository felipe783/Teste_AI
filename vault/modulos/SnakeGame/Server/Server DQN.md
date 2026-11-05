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

Desde 2026-11-05 todos os valores acima são **padrões** de um `Config` (dataclass em `agent.py`) e podem ser mudados pela CLI (`python DQN/agent.py --help`) sem editar o código. A execução também ganhou `--n-step` (retornos de n passos, padrão 1) e `--safe-explore` (exploração só entre ações que não colidem, desligada). Cada execução tem um `--run-dir` com `config.json`, `Models/` e `logs/`; ao retomar, o que não for passado na CLI vem do `config.json`. Ver [[Correções Snake DQN 2026-11-05]].
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

## Checkpoint e Log (desde 2026-11-05)

Caminhos relativos ao `--run-dir` (padrão `.`, o layout antigo):
- `Models/model.pth`: **último** estado. Leva pesos, *target*, otimizador, `n_game`, `record`, `train_steps`, `env_steps`, `train_seconds`, `best_eval_mean`, `eval_protocol`, `config`, `git_commit`, `saved_at` e `checkpoint_version = 3`. É salvo a cada 100 partidas **ou** 10 minutos, a cada recorde e na parada (`SIGTERM` ou `Ctrl+C`).
- Gravação **atômica** (`.tmp` → `fsync` → `os.replace`), com a versão anterior em `*.bak`. Ao retomar, um arquivo ilegível cai no `.bak`; sem `.bak` válido, ou com outro `STATE_SIZE`, o treino **para com código 2 sem sobrescrever nada** (antes recomeçava do zero em silêncio).
- `Models/best_model.pth`: **melhor** modelo pela avaliação gulosa (ver [[#Regressão 15k × 30k (2026-10-17)]]). O `best_eval_mean` de referência é lido **do próprio arquivo** ao iniciar, não do `model.pth` (DQN-01); se ele foi medido com outro protocolo (nº de partidas ou semente), é reavaliado.
- `Models/replay.npz`: replay buffer (~18 MB com 100 mil transições), salvo nos checkpoints periódicos. Sem ele, a retomada coleta 20 000 transições antes de treinar.
- `logs/eval_log.csv`: `partida, media, ic95, mediana, min, max, melhor_media, loss_media, epsilon, train_steps, passos_ambiente, horas_treino, passos_por_s`. Um CSV com o cabeçalho antigo é arquivado.
- `TrainingLogger` grava `logs/training_log.csv` (`partida, score, max, epsilon`) e **rotaciona** (renomeia) o arquivo aos 256 MB.
- O `training_log.csv` subestima o agente: com ε = 0,02, o modelo de 62 200 partidas faz ~31 pontos, contra ~101 guloso, porque as ações aleatórias matam a cobra longa. Use o `eval_log.csv` para acompanhar.

## Avaliação

````bash
python DQN/agent.py --eval --episodes 20
python DQN/agent.py --eval --episodes 50 --model Models/m15k.pth Models/m30k.pth   # comparação pareada
python DQN/agent.py --games 15000 --seed 0                                           # treino com orçamento e semente
````

Roda sem exploração (`explore=False`). A avaliação de seleção do treino usa 50 partidas desde 2026-11-05 (antes 20, IC ±13). A partida *i* usa `random.seed(seed + i)` (padrão `eval_seed = 12345`), então todos os modelos partem da mesma sequência aleatória e, como a política gulosa é determinística, o resultado de um checkpoint é reprodutível. As comidas só coincidem no começo: o sorteio depende das células livres, e em 2026-11-03 dois modelos divergiram na 7ª comida (DQN-04 da [[Auditoria Snake DQN 2026-11-03]]). Imprime média ± IC95%, mediana, mínimo e máximo; com dois modelos, imprime a **diferença pareada** por partida e se ela é significativa.

> [!note] Desempenho
> Em 2026-11-05 o ambiente passou a usar BFS em grade de inteiros com cache (5 → 3 BFS por passo), e o replay virou arrays NumPy: o treino foi de 528 para **2 395 passos/s** (1 thread), com placares idênticos ao código antigo no mesmo modelo. Para treinar em paralelo, rode vários processos de 1 thread ([[Treino na AWS]]).

> [!note] Modelo treinado
> Em 2026-11-03, `Test/Models/model.pth` (não versionado) tinha `n_game = 62200`, `record = 122`, `train_steps = 10 562 997`, `train_every_steps = 4` e `best_eval_mean = 109,15`, ou seja, um treino longo já com o código de 2026-10-17. Também estavam lá `model10k.pth` (8 400 partidas), `model30k.pth` (37 000) e os antigos `model15k/20K/25k/30k/35K-Antigo.pth`. Avaliação pareada com 40 partidas e `--seed 777`: `model.pth` 101,0 ± 10,3 · `model30k.pth` 101,5 ± 9,0 · `model10k.pth` 83,2 ± 8,9 · `model30k-Antigo.pth` 81,6 ± 10,2 (platô desde ~37 000 partidas). Ver [[Auditoria Snake DQN 2026-11-03]].
> Em 2026-10-17 o mesmo arquivo tinha 14 500 partidas; o `model_antigo.pth` de 6 400 partidas citado abaixo não está mais nesta máquina (o `Server_DQN/Models/model_antigo.pth` local é um `state_dict` de 11 entradas, da v1).

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
2. Treinar no servidor do zero com o código novo (fazer backup de `Models/` e `logs/` antes) e acompanhar `logs/eval_log.csv` até 30 000 partidas. **Feito** (passou de 62 000); o `eval_log.csv` ficou no servidor. A avaliação local de 2026-11-03 mostra platô em ~101 pontos desde ~37 000 partidas e que o código novo supera o modelo antigo de 30k em +19,4 ± 14,5 ([[Auditoria Snake DQN 2026-11-03]]).
3. Comparar `best_model.pth` com o modelo antigo de 30k: `python DQN/agent.py --eval --episodes 50 --seed 777 --model <antigo> Models/best_model.pth`.
4. Não alterados: a v1 (`SnakeGame/DQN`) tinha o mesmo treino online e foi removida em 2026-10-29; a cópia de `snake_gameai.py` em `Test/` não recebeu a otimização do BFS.
