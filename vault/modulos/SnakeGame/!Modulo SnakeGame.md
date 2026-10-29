O módulo **`SnakeGame`** reúne os experimentos de **IA jogando Snake**. Todos compartilham a mesma ideia: uma rede `N entradas → 256 → 3 saídas` escolhe entre seguir reto, virar à direita ou virar à esquerda.

Submódulos:
- **[[!Modulo DQN|DQN]]**
    - Aprendizado por reforço (Deep Q-Network) local, com janela pygame e vídeos. **Código removido em 2026-10-29**; as notas ficam como histórico.
- **[[!Modulo GA|GA]]**
    - Neuroevolução: população de redes evoluída por seleção, crossover e mutação. Desde 2026-10-29, o código só existe em `Server/Server_GA/`.
- **[[!Modulo Server|Server]]**
    - Versões headless em Docker para treino contínuo; inclui o **DQN v2**.
- **[[!Modulo Test|Test]]**
    - Carrega um modelo treinado, assiste às partidas e grava os recordes.
- **[[Ambiente Snake]]**
    - O jogo em si, que existe em três variações.

> [!note] Execução
> Os scripts importam módulos irmãos sem pacote (`from snake_gameai import ...`, `from model import ...`). Rode cada um **de dentro da sua pasta**.
