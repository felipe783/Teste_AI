### Linguagem Principal

``Python``
### Subprojetos

O repositório tem dois subprojetos independentes, que não compartilham código:
- **SnakeGame**: IA que aprende a jogar Snake.
- **Sudoku**: gerador e resolvedor de Sudoku.

### SnakeGame

A IA observa o jogo como um **vetor de estado** (perigos ao redor, direção, posição da comida…) e escolhe uma de três ações: `[1,0,0]` seguir reto, `[0,1,0]` virar à direita, `[0,0,1]` virar à esquerda.

A decisão é tomada por uma rede `Linear_QNet` com uma camada oculta de 256 neurônios. Ela é treinada de duas formas:

- **DQN (Deep Q-Network)**: a rede aprende a estimar a recompensa futura de cada ação (valor Q) a partir da experiência acumulada.
> [[Agente DQN]] (v1, 11 entradas) · [[Server DQN]] (v2, 21 entradas, Double DQN)
- **Algoritmo Genético**: uma população de redes joga, as melhores são selecionadas, cruzadas e mutadas, sem gradiente.
> [[Agente Genético]] · [[Treino Genético]]

As recompensas base são **+10** ao comer e **−10** ao morrer (colisão ou partida longa demais: `frame_iteration > 100 * len(snake)`). A v2 adiciona *reward shaping*.

### Sudoku

Gera um tabuleiro completo por Backtracking randomizado, remove números conforme a dificuldade e tenta resolver com o método escolhido.
- **Tradicional NxN**: qualquer quadrado perfeito (4, 9, 16…).
> [[Geração do Tabuleiro]] · [[Resolvedores]]
- **Samurai 33x33**: 9 tabuleiros 9x9 sobrepostos, resolvido só por Backtracking.
> [[Sudoku Samurai]]

## Fluxo Geral — Sudoku

1. O usuário escolhe o tipo, o tamanho, a quantidade de jogos, a dificuldade e o método ([[Main Sudoku]]).
2. `setupGame` define quantas células ficam reveladas (50%, 40% ou 30%).
3. `generateSudoku` (ou `generateSamurai`) gera a solução e remove números.
4. O resolvedor escolhido preenche o tabuleiro.
5. São calculados quadrantes corretos, cobertura e status.
6. Ao final, o resumo é exibido e acrescentado em `results.json` ([[Logs Sudoku]]).

## Fluxo Geral — Treino DQN

1. O agente lê o estado do jogo (`get_state`).
2. Escolhe a ação: aleatória com probabilidade ε (exploração) ou `argmax` da rede (aproveitamento).
3. `play_step` executa a ação e devolve `reward, done, score`.
4. A transição é usada no treino imediato (*short memory*) e guardada no *replay buffer*.
5. Ao fim da partida, um lote aleatório do buffer treina a rede (*long memory*), e checkpoint/log/vídeo são salvos conforme as regras.

## Fluxo Geral — Treino Genético

1. Uma população de 80 agentes (ou clones do melhor checkpoint) é criada.
2. Cada agente joga uma partida headless e recebe `fitness = score² × 100 + passos`.
3. O melhor da geração é salvo se bater o recorde, e a geração é registrada em `logs/history.json`.
4. A próxima geração mantém a elite (15%) e completa a população com filhos de torneio + crossover + mutação.
