# 🐍 SnakeGame AI

Uma rede neural aprende a jogar Snake sozinha. A rede `entradas → 256 → 3` observa o jogo (perigos ao redor, direção, posição da comida…) e escolhe entre **seguir reto**, **virar à direita** ou **virar à esquerda**. Ela é treinada de duas formas, para comparar as abordagens:

- **DQN (Deep Q-Network)**: aprendizado por reforço com *replay buffer* e política ε-greedy.
- **Algoritmo Genético (GA)**: uma população de redes evolui por seleção por torneio, crossover, elitismo e mutação auto-adaptativa, sem gradiente.

## 📁 Pastas

| Pasta | Conteúdo |
|---|---|
| `DQN/` | DQN v1: treino local com janela pygame, gráfico e vídeos (11 entradas) |
| `GA/` | Treino genético local, headless, com gráfico (15 entradas) |
| `Server/Server_DQN/` | **DQN v2** para Docker: Double DQN, *target network*, estado de 21 entradas com BFS, *reward shaping*. Ver [Server/README.md](Server/README.md) |
| `Server/Server_GA/` | Versão do GA para Docker (em ajuste, ainda não executa) |
| `Test/` | Assiste um modelo treinado jogar e grava em vídeo só as partidas recordistas |

## 🚀 Como rodar

Na raiz do repositório, com o ambiente ativado:

```bash
pip install -r SnakeGame/requirements.txt
```

Os scripts importam os arquivos vizinhos diretamente, então **rode cada um de dentro da sua pasta**:

```bash
# Treino DQN v1 (abre a janela do jogo; Ctrl+C salva o checkpoint)
cd SnakeGame/DQN
python agent.py

# Treino genético (Ctrl+C salva o melhor indivíduo)
cd SnakeGame/GA
python train_genetic.py

# DQN v2 sem Docker: treino e avaliação
cd SnakeGame/Server/Server_DQN
python DQN/agent.py
python DQN/agent.py --eval --episodes 20

# Assistir o modelo DQN v2 jogar (precisa de Test/Models/model.pth)
cd SnakeGame/Test
python play.py
```

### Onde ficam os resultados

| Script | Saída |
|---|---|
| `DQN/agent.py` | `DQN/models/checkpoint.pth`, `videos/*.mp4` |
| `GA/train_genetic.py` | `GA/Models/best_genetic.pth`, `GA/Models/generation_XXX.pth`, `logs/history.json` |
| `Server_DQN/DQN/agent.py` | `Models/model.pth`, `logs/training_log.csv` |
| `Test/play.py` | `Test/video/*.mp4`, `Test/logs/testes.json` |

> `Models/`, `logs/`, `*.json` e `*.mp4` estão no `.gitignore`: um clone novo **não traz** os modelos treinados do DQN v2. Copie o `model.pth` do servidor para `Test/Models/` antes de rodar o `play.py`.
> A opção 2 do `play.py` (modelo genético) ainda não é compatível com a rede do GA.

## 📚 Documentação

A documentação completa do projeto está em [`vault/_Teste_AI.md`](../vault/_Teste_AI.md). Comece por esse arquivo para consultar a arquitetura, os algoritmos, as decisões técnicas, os problemas conhecidos e os próximos passos. Os detalhes deste subprojeto estão em [`vault/modulos/SnakeGame/`](<../vault/modulos/SnakeGame/!Modulo SnakeGame.md>).
