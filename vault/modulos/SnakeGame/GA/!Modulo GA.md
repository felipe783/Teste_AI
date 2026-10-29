> [!warning] Versão local removida em 2026-10-29
> A pasta `SnakeGame/GA/` foi apagada no commit `0a73702`. O algoritmo descrito aqui continua só na cópia `SnakeGame/Server/Server_GA/GA/`, com as diferenças listadas em [[Server GA]], e essa cópia **não executa** no estado atual. Para ler o código antigo: `git show dfbc504:SnakeGame/GA/<arquivo>`.

O módulo **`GA`** treina a rede por **Neuroevolução**: em vez de gradiente, uma população de redes joga, e os pesos das melhores são recombinados e mutados geração após geração.

Normalmente, um Algoritmo Genético contém:
- **Indivíduo / Genoma**
    - Aqui, o vetor achatado de todos os pesos da rede. Ver [[Rede Genética]].
- **Função de Fitness**
    - Mede o desempenho de cada indivíduo. Ver [[Agente Genético]].
- **Seleção**
    - Escolhe os pais (torneio) e preserva os melhores (elitismo). Ver [[Treino Genético]].
- **Crossover e Mutação**
    - Combinam e perturbam os genomas para explorar novas soluções.

O jogo roda **headless** (`Server_GA/GA/snake_gameai.py`, sem pygame), o que permite avaliar 80 indivíduos por geração rapidamente. Ver [[Ambiente Snake]].
