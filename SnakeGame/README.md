# 🐍 SnakeGame AI

Uma rede neural aprende a jogar Snake sozinha. A rede `entradas → 256 → 3` observa o jogo (perigos ao redor, direção, posição da comida…) e escolhe entre **seguir reto**, **virar à direita** ou **virar à esquerda**. Ela é treinada de duas formas, para comparar as abordagens:

- **DQN (Deep Q-Network)**: aprendizado por reforço com *replay buffer* e política ε-greedy.
- **Algoritmo Genético (GA)**: uma população de redes evolui por seleção por torneio, crossover, elitismo e mutação auto-adaptativa, sem gradiente.

## 📁 Pastas

| Pasta | Conteúdo |
|---|---|
| `Server/Server_DQN/` | **DQN v2** para Docker: Double DQN, *target network*, estado de 21 entradas com BFS, *reward shaping*. Ver [Server/README.md](Server/README.md) |
| `Server/Server_GA/` | Única versão do GA, para Docker (15 entradas; em ajuste, ainda não executa) |
| `Test/` | Assiste um modelo treinado jogar e grava em vídeo só as partidas recordistas |

## 🚀 Como rodar

Na raiz do repositório, com o ambiente ativado:

```bash
pip install -r SnakeGame/requirements.txt
```

Os scripts importam os arquivos vizinhos diretamente, então **rode cada um de dentro da sua pasta**:

```bash
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
| `Server_DQN/DQN/agent.py` | `Models/model.pth`, `Models/best_model.pth`, `logs/training_log.csv`, `logs/eval_log.csv` |
| `Test/play.py` | `Test/video/*.mp4`, `Test/logs/testes.json` |

> `Models/`, `logs/`, `*.json` e `*.mp4` estão no `.gitignore`: um clone novo **não traz** os modelos treinados do DQN v2. Copie o `model.pth` do servidor para `Test/Models/` antes de rodar o `play.py`.
> A opção 2 do `play.py` (modelo genético) ainda não é compatível com a rede do GA.
> O DQN v1 (`DQN/`) e o GA local (`GA/`) foram removidos em 2026-10-29. O código antigo continua no histórico do Git (`git show dfbc504:SnakeGame/DQN/agent.py`).

## 📚 Documentação

A documentação completa do projeto está em [`vault/_Teste_AI.md`](../vault/_Teste_AI.md). Comece por esse arquivo para consultar a arquitetura, os algoritmos, as decisões técnicas, os problemas conhecidos e os próximos passos. Os detalhes deste subprojeto estão em [`vault/modulos/SnakeGame/`](<../vault/modulos/SnakeGame/!Modulo SnakeGame.md>).
