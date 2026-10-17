# Visão Geral do Sistema

O Teste_AI é um repositório de experimentos com IA aplicada a jogos. O SnakeGame treina uma rede neural para jogar Snake usando [[!Modulo DQN|DQN]] e [[!Modulo GA|Algoritmo Genético]]; o Sudoku gera e resolve tabuleiros com algoritmos clássicos de busca.

## Finalidade

- Comparar **Aprendizado por Reforço** (DQN) e **Algoritmo Genético** sobre a mesma tarefa (Snake).
- Treinar por longos períodos em servidor (Docker) e avaliar visualmente os modelos depois.
- Comparar estratégias de resolução de Sudoku: probabilidade ponderada, tentativas repetidas e Backtracking.
- Registrar métricas (scores, recordes, cobertura, tempo) para análise.

## Entrada principal

- [[Main Sudoku|Sudoku/main.py]]: menu interativo no terminal.
- [[Agente DQN|SnakeGame/DQN/agent.py]] e [[Treino Genético|SnakeGame/GA/train_genetic.py]]: treinos locais.
- [[Server DQN|Server_DQN/DQN/agent.py]]: treino/avaliação no servidor (`--eval`).
- [[Play|SnakeGame/Test/play.py]]: execução e gravação dos modelos treinados.
## Saída principal

- Sudoku: tabuleiros coloridos (ANSI) e `results.json` ([[Logs Sudoku]]).
- SnakeGame: checkpoints `.pth`, `logs/history.json` (GA), `logs/training_log.csv` (DQN v2), `logs/testes.json` e vídeos `.mp4` ([[Play]]).

## Módulos principais

- [[!Modulo Sudoku]]: geração, resolução, Samurai e métricas.
- [[!Modulo SnakeGame]]: [[Ambiente Snake]] e os quatro submódulos abaixo.
- [[!Modulo DQN]]: Deep Q-Network v1, local, com pygame.
- [[!Modulo GA]]: neuroevolução com mutação auto-adaptativa.
- [[!Modulo Server]]: versões headless para Docker (DQN v2 e GA).
- [[!Modulo Test]]: avaliação visual com gravação de recordes.
- [[!Modulo Ferramentas]]: inspeção de checkpoints.


[[Dependencias (Teste_AI)]]
