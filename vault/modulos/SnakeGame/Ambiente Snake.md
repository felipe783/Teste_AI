# Sobre os Arquivos

Os arquivos `snake_gameai.py` implementam o **Ambiente do Jogo**: estado da cobra, comida, movimento, colisões e recompensa. É o "mundo" com o qual o agente interage através de `reset()` e `play_step(action)`.
# Funcionalidades Comuns

## Ação Relativa

A ação é **relativa à direção atual**, o que impede a cobra de "voltar" sobre si mesma:

````python
clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
idx = clock_wise.index(self.direction)
# [1,0,0] reto | [0,1,0] direita (idx + 1) | [0,0,1] esquerda (idx - 1)
````

## play_step

1. Move a cabeça e a insere no início de `snake`.
2. Se colidiu com parede/corpo, ou se `frame_iteration > 100 * len(snake)` (anti-loop) → `reward = -10`, `game_over = True`.
3. Se comeu → `score += 1`, `reward = 10`, nova comida. Senão remove a cauda (`snake.pop()`).
4. Retorna `reward, game_over, score`.

- Tabuleiro: 640x480 px, `BLOCK_SIZE = 20` → grade 32x24 = 768 células.

# As Quatro Versões

| Arquivo | Interface | Extras | Usado por |
|---|---|---|---|
| `DQN/snake_gameai.py` | pygame, `SPEED = 60` | Captura de frames e `save_video` (OpenCV) | [[Agente DQN]] |
| `GA/snake_gameai.py` (= `Server_GA/GA/`) | Headless | — | [[Agente Genético]] |
| `Server_DQN/DQN/snake_gameai.py` | Headless | `action_analysis` (BFS), comida só em células livres, *reward shaping* | [[Server DQN]] |
| `Test/snake_gameai.py` | pygame, `SPEED = 40` | Mesma lógica do Server DQN + desenho | [[Play]] |

## action_analysis (versão Server DQN / Test)

Simula cada uma das 3 ações e devolve `(colisão, fração de espaço alcançável, comida alcançável, cauda alcançável)`. O espaço é calculado por **BFS** (`_reachable_area`), tratando o corpo como parede, menos a cauda, que vai se mover.

## Reward Shaping (versão Server DQN)

````python
progress = (old_distance - new_distance) / BLOCK_SIZE
reward = -0.01 + 0.10 * progress + 0.05 * (new_space - old_space)
````

- `-0.01` por passo desestimula andar em círculos.
- `+0.10 × progresso` recompensa aproximar-se da comida (distância Manhattan).
- `+0.05 × Δespaço` recompensa manter área livre, para evitar se encurralar.

> [!note] Duplicação e memória
> As quatro versões divergiram por cópia. Na versão DQN, `capture_frame()` acumula **todos** os frames da partida em `video_frames`, mesmo quando o vídeo não é salvo, e `_update_ui()` é chamado duas vezes por passo.
