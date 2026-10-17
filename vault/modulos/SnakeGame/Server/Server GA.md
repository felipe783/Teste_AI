# Sobre o Arquivo

Os arquivos de `Server/Server_GA/GA/` são uma **cópia adaptada do [[!Modulo GA|GA]]** para rodar no servidor: sem gráfico e com população inicial sempre aleatória.

Diferenças em relação a `SnakeGame/GA/`:
- Sem `Helper.plot` (headless).
- `ELITE_FRACTION = 0.10` (8 elites, em vez de 12).
- População inicial sempre `Agent()` novo; o checkpoint carregado entra só na posição `0`.
- Limite de passos sem comida fixo em `100`.
- `history.json` gravado diretamente, sem o `.tmp` + `os.replace`.

> [!warning] Não executa no estado atual
> - **Imports inexistentes**: `from Teste_AI.SnakeGame.Server.GA_Server.GA.agent import Agent`. A pasta se chama `Server_GA`, e dentro do container não existe o pacote `Teste_AI`, então o resultado é `ModuleNotFoundError`.
> - O `Dockerfile` executa `DQN/agent.py`, que não existe nessa imagem.
> - Os volumes do `run_docker.sh` apontam para `/Treinamento_DQN/...`, mas o `WORKDIR` é `/Treinamento_GA`. Além disso, os modelos são salvos em `GA/Models` (via `__file__`), fora do volume.
> Ver [[Docker]].
