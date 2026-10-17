# Estrutura do Projeto

```text
Teste_AI/
	📁 SnakeGame
	|  |--📁 DQN                 ← DQN v1 (pygame + vídeo)
	|  |  |--📁 models
	|  |  |  |--checkpoint.pth   ← versionado (11 entradas, recorde 120)
	|  |  |  |--model.pth        ← state_dict puro (formato antigo)
	|  |  |--agent.py
	|  |  |--Helper.py
	|  |  |--model.py
	|  |  |--snake_gameai.py
	|  |--📁 GA                  ← Algoritmo Genético
	|  |  |--__init__.py
	|  |  |--agent.py
	|  |  |--Helper.py
	|  |  |--model.py
	|  |  |--snake_gameai.py
	|  |  |--train_genetic.py
	|  |--📁 Server              ← versões para Docker
	|  |  |--📁 Server_DQN
	|  |  |  |--📁 DQN
	|  |  |  |  |--agent.py      ← DQN v2 (Double DQN, 21 entradas)
	|  |  |  |  |--model.py
	|  |  |  |  |--snake_gameai.py
	|  |  |  |--Dockerfile
	|  |  |  |--requirements.txt
	|  |  |  |--run_docker.sh
	|  |  |--README.md
	|  |  |--📁 Server_GA
	|  |     |--📁 GA
	|  |     |  |--agent.py
	|  |     |  |--model.py
	|  |     |  |--snake_gameai.py
	|  |     |  |--train_genetic.py
	|  |     |--Dockerfile
	|  |     |--requirements.txt
	|  |     |--run_docker.sh
	|  |--📁 Test                ← avaliação visual
	|  |  |--📁 Models           ← não versionado (model.pth, model_antigo.pth)
	|  |  |--📁 logs             ← não versionado (testes.json)
	|  |  |--📁 video            ← não versionado (recordes .mp4)
	|  |  |--model.py
	|  |  |--play.py
	|  |  |--snake_gameai.py
	|  |--requirements.txt
	|  |--README.md
	📁 Sudoku
	|  |--📁 core
	|  |  |--__init__.py
	|  |  |--config.py           ← posições dos 9 tabuleiros do Samurai
	|  |--📁 src
	|  |  |--📁 SamuraiGame
	|  |  |  |--__init__.py
	|  |  |  |--BoardSamuraiGeneration.py
	|  |  |  |--SamuraiBoard.py
	|  |  |--Board.py
	|  |  |--BoardGeneration.py
	|  |  |--gameConfig.py
	|  |  |--Logs.py
	|  |  |--Probability.py
	|  |--image.png              ← layout do Samurai (usado no README)
	|  |--main.py
	|  |--README.md
	|  |--results.json           ← não versionado (histórico de execuções)
	📁 vault                     ← esta documentação (Obsidian); índice em _Teste_AI
	|--leitor_pth.py             ← inspeção de checkpoints
	|--test.py                   ← experimento de shape de tensor 3D
	|--README.md
	|--.gitignore
```
## Pastas

- `SnakeGame/DQN/`: treino DQN local com janela. Ver [[!Modulo DQN]].
- `SnakeGame/GA/`: treino genético local. Ver [[!Modulo GA]].
- `SnakeGame/Server/`: imagens Docker para treino contínuo. Ver [[!Modulo Server]].
- `SnakeGame/Test/`: execução dos modelos e gravação de vídeos. Ver [[!Modulo Test]].
- `Sudoku/`: gerador e resolvedor. Ver [[!Modulo Sudoku]].
- Raiz: utilitários. Ver [[!Modulo Ferramentas]].
