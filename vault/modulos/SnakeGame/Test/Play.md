# Sobre o Arquivo

O arquivo `play.py` é o **Executor de Testes**: carrega o checkpoint escolhido, joga as partidas pedidas, grava vídeos dos recordes e registra um resumo em `logs/testes.json`.
# Funcionalidades

## Escolha do Modelo

````python
if escolha == 1:
    MODEL_PATH = "Models/model.pth"          # DQN v2 (21 entradas)
elif escolha == 2:
    MODEL_PATH = "Models/best_genetic.pth"   # GA
````

O `get_state` é o mesmo de 21 entradas do [[Server DQN]], e a rede é criada como `Linear_QNet(21, 256, 3)`.

## Gravação Condicional

- Cada partida é gravada num arquivo temporário `video/.temporario_<timestamp>.mp4`.
- Ao fim, se `score > recorde_atual`, o arquivo é renomeado para `snake_<timestamp>_score_<N>.mp4`; senão é apagado.
- O recorde inicial é o maior entre os scores nos nomes dos vídeos existentes e os `recorde_final` dos logs.

## Histórico

`salvar_log` acrescenta `teste_N` em `logs/testes.json` com modelo, jogos pedidos/rodados, média, mínimo, máximo, recordes e vídeos salvos. Fica no `finally`, então é salvo mesmo com `Ctrl+C`.

````json
"teste_1": {
  "modelo": "Models/model.pth",
  "jogos_solicitados": 20,
  "jogos_rodados": 1,
  "maior_pontuacao": 31,
  "recorde_final": 31,
  "videos_salvos": ["snake_2026-09-11_09-00-38_434018_score_31.mp4"]
}
````

> [!warning] Problemas
> - **Opção 2 incompatível**: o GA salva redes de 15 entradas, e o `play.py` cria uma de 21, o que daria erro de *shape* no `load_state_dict`. Além disso, não há `best_genetic.pth` em `Test/Models` (verificado em 2026-10-17).
> - **Primeiro uso**: sem `logs/testes.json`, `carregar_testes()` retorna `[]` (lista), e `obter_recorde_anterior()` chama `.values()` → `AttributeError`. Como `logs/` está no `.gitignore`, isso acontece em todo clone novo.
> - `MODEL_PATH` é relativo: rode de dentro de `SnakeGame/Test/`.
