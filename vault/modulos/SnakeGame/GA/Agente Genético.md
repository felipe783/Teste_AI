# Sobre o Arquivo

> [!warning] Versão local removida em 2026-10-29
> A pasta `SnakeGame/GA/` foi apagada no commit `0a73702`. O algoritmo descrito aqui continua só na cópia `SnakeGame/Server/Server_GA/GA/`, com as diferenças listadas em [[Server GA]], e essa cópia **não executa** no estado atual. Para ler o código antigo: `git show dfbc504:SnakeGame/GA/<arquivo>`.

O arquivo `agent.py` define o **Indivíduo da População**: a rede, o estado de entrada, a avaliação por fitness, os operadores genéticos e o log por geração.
# Funcionalidades

## get_state (15 entradas)

As 11 entradas do [[Agente DQN|DQN v1]] (removido), mais:
- **perigo a 2 blocos** (reto, direita, esquerda), para enxergar além da célula imediata;
- **tamanho relativo da cobra**: `len(snake) / 768`.

## play_and_evaluate (Fitness)

Joga uma partida completa sem exploração aleatória (`argmax` direto):

````python
if steps_since_food > max_steps_without_food:  # Evitar LOOP
    done = True
if done:
    self.fitness = (score ** 2) * 100 + steps
````

- `score²` faz cada comida valer cada vez mais; `steps` desempata e premia sobreviver.
- O limite de passos sem comer começa em `max(100, 20 × len(snake))` e volta a `100` depois de comer.

## crossover (uniforme)

Para cada peso, sorteia se vem do pai A ou do B (`mask = rand < 0.5`). Os parâmetros de mutação do filho são a média dos pais.

## mutate (auto-adaptativa)

Cada indivíduo carrega a própria `mutation_rate` e `mutation_strength`, que também evoluem por um fator log-normal:

````python
self.mutation_strength *= math.exp(SELF_ADAPT_TAU * np.random.randn())
self.mutation_rate     *= math.exp(SELF_ADAPT_TAU * np.random.randn())
# limitados a [0.01, 2.0] e [0.01, 0.9]

mask = torch.rand_like(weights) < self.mutation_rate
noise = torch.randn_like(weights) * self.mutation_strength
weights[mask] += noise[mask]
````

- Valores iniciais: `MUTATION_RATE = 0.15`, `MUTATION_STRENGTH = 0.30`, `SELF_ADAPT_TAU = 0.1`.

## save_generation_log

Acrescenta a geração em `logs/history.json` (relativo ao diretório atual), com dados de **cada indivíduo**. A escrita é atômica: grava em `.tmp` e faz `os.replace`, o que evita corromper o histórico se o processo morrer no meio.

> [!note] Crescimento do log
> O histórico é relido e reescrito inteiro a cada geração, com 80 indivíduos por entrada. O custo cresce de forma linear com o número de gerações.
