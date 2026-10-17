# 🐳 SnakeGame — Treino em Servidor

Versões do treino **sem interface gráfica**, empacotadas em Docker para rodar continuamente num servidor Linux. Os modelos e logs ficam em volumes do host, então o treino continua de onde parou mesmo que o container seja recriado.

| Pasta | Algoritmo | Estado |
|---|---|---|
| `Server_DQN/` | **DQN v2**: Double DQN, *target network* (atualizada a cada 1 000 passos), Huber loss, gradient clipping, ε linear de 1.0 a 0.02, estado de 21 entradas com análise de espaço livre por BFS | Funcional |
| `Server_GA/` | Algoritmo Genético (cópia adaptada de `../GA`) | **Não executa ainda**: imports, `CMD` do Dockerfile e volumes precisam de ajuste |

## 🚀 Como rodar

### Com Docker (servidor Linux)

```bash
bash SnakeGame/Server/Server_DQN/run_docker.sh
```

O script:
1. constrói a imagem `snake_dqn`;
2. remove o container `Snake_DQN` antigo, se existir;
3. cria o container com `--restart unless-stopped` e os volumes
   `/home/felipe/Teste_IA/Server_DQN/Models` → `/Treinamento_DQN/Models` e
   `/home/felipe/Teste_IA/Server_DQN/logs` → `/Treinamento_DQN/logs`;
4. inicia o treino.

> Ajuste `BASE_DIR` no `run_docker.sh` se o servidor usar outro caminho.

Acompanhar:

```bash
docker logs -f Snake_DQN
tail -f /home/felipe/Teste_IA/Server_DQN/logs/training_log.csv
```

### Sem Docker

```bash
cd SnakeGame/Server/Server_DQN
pip install -r requirements.txt
python DQN/agent.py                         # treina
python DQN/agent.py --eval --episodes 100   # avalia sem explorar nem salvar
```

### Levar o modelo para assistir

Copie `Models/model.pth` do servidor para `SnakeGame/Test/Models/` e rode `python play.py` dentro de `SnakeGame/Test`.

## 📚 Documentação

A documentação completa (features do estado, *reward shaping*, Double DQN, Docker e os problemas do Server GA) está no **Obsidian de contextos gerais**, no projeto **Teste_AI**, módulo Server. Ela também é versionada em [`../../vault/`](../../vault/_Teste_AI.md).
