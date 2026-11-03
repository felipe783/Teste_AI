# Auditoria técnica do Snake DQN — 2026-11-03

Escopo: o **DQN v2** em `SnakeGame/Server/Server_DQN/` ([[Server DQN]], [[Ambiente Snake]], [[Docker]]) e o `SnakeGame/Test/play.py` ([[Play]]), que consome os modelos dele. "Snake_DQN" é só o nome do container Docker; não existe pasta com esse nome.

A auditoria foi feita **sem alterar código nem modelos**. Os experimentos rodaram com `python -B` em diretório temporário ou em cópias, e os checkpoints de `Test/Models/` foram apenas lidos. Este relatório é histórico: os itens resolvidos depois devem ser marcados no índice [[_Teste_AI]], não reescritos aqui.

Marcadores: **[CONFIRMADO 2026-11-03]** executado ou medido · **[LIDO 2026-11-03]** conclusão tirada da leitura do código, sem execução · **[PROVÁVEL]** dedução não testada.

## 1. Resumo executivo

- O núcleo do DQN está **correto**. Double DQN, target network, Huber, gradient clipping, máscara `~done`, treino só pelo replay e seleção do modelo por avaliação gulosa: não há erro matemático no `train_step`.
- **O agente estagnou em cerca de 100 pontos**, uns 13% do tabuleiro de 768 células. [CONFIRMADO 2026-11-03] 40 partidas gulosas por checkpoint, sementes 777+i (nunca usadas na seleção):

| Checkpoint (`Test/Models/`) | Partidas de treino | Média ± IC95% | Mediana | Diferença pareada vs. `model.pth` |
|---|---|---|---|---|
| `model.pth` (código novo) | 62 200 | **101,0 ± 10,3** | 104 | — |
| `model30k.pth` (código novo) | 37 000 | **101,5 ± 9,0** | 100,5 | −0,6 ± 10,4 (não significativa) |
| `model10k.pth` (código novo) | 8 400 | 83,2 ± 8,9 | 81 | +17,8 ± 11,3 |
| `model30k-Antigo.pth` (código antigo) | 29 900 | 81,6 ± 10,2 | 80,5 | +19,4 ± 14,5 |

- **O que a tabela mostra:** a mudança de 2026-10-17 (lote do replay a cada 4 passos) rendeu cerca de 20 pontos. De 37 mil a 62 mil partidas, o ganho foi **zero**.
- **O limite não está no DQN**, e sim em três escolhas do projeto: o horizonte curto (γ = 0,95), uma recompensa que pune o desvio seguro e uma representação que não deixa explícito o que é armadilha.
- **Dois problemas de engenharia graves, reproduzidos:** sem `model.pth`, o `best_model.pth` é destruído (DQN-01). Com `model.pth` truncado, o treino cai e o Docker o reinicia em loop (DQN-02).

## 2. Arquitetura e fluxo

```
SnakeGameAI (snake_gameai.py)        Agent (agent.py)                    QTrainer (model.py)
 reset / play_step(action)  ──►  get_state: 21 features (3 BFS)  ──►  Linear 21→256→ReLU→3
 action_analysis (BFS)            get_action: ε-greedy                 Double DQN + target net
 reward shaping                   remember → deque(100k)                Huber, Adam, clip 10
                                  train(): a cada 4 passos ──────────►  train_step(lote de 256)
                                  a cada 500 partidas: run_eval(20, seed 12345) → best_model.pth
```

**Ambiente**
- Grade de 32×24 com `BLOCK_SIZE = 20`. A comida só nasce em célula livre.
- A partida acaba por colisão, quando `frame_iteration > 100 × len(snake)` ou quando o tabuleiro enche.
- São 3 ações relativas à direção atual: reto, direita e esquerda.

**Estado (21 features)**
- Direção atual, absoluta e em one-hot: 4.
- dx e dy até a comida e até a cauda, absolutos e normalizados: 4.
- Para cada ação: colisão, fração do tabuleiro alcançável por BFS, comida alcançável e cauda alcançável: 12.
- Comprimento da cobra / 768: 1.

**Recompensa**
- +10 ao comer e −10 ao morrer ou estourar o timeout.
- Nos outros passos: `−0,01 + 0,10·progresso + 0,05·Δespaço`.

**Aprendizado**
- 1 passo de gradiente com lote de 256 a cada 4 passos do ambiente.
- A target network é sincronizada a cada 250 passos de gradiente.
- ε decai linearmente de 1,0 a 0,02 em 62 500 passos de gradiente.

**Persistência**
- `model.pth` (último estado) a cada 100 partidas, a cada recorde e no Ctrl+C.
- `best_model.pth` quando a média da avaliação melhora.
- `training_log.csv` e `eval_log.csv`.

**Inferência:** o `play.py` reimplementa o `get_state` (é uma 4ª cópia) e carrega o `model.pth`, não o `best_model.pth`.

## 3. Avaliação da implementação do DQN

| Item | Veredito | Raciocínio |
|---|---|---|
| Alvo de Bellman / Double DQN | ✅ Correto | A rede principal escolhe a ação e a target a avalia. O `~done` zera o bootstrap; `no_grad` no alvo (`model.py:69-75`). |
| Target network | ✅ | `requires_grad=False`, `eval()`, cópia por `load_state_dict`; coberta por teste. |
| Loss e otimizador | ✅ | Huber com δ = 1 é adequado para ±10. Adam com LR 2,5e-4 e clip 10 são padrão. |
| ε-greedy | ✅ com ressalva | Chega ao mínimo nas primeiras ~1–2 mil partidas (250 mil passos do ambiente). Daí em diante são 2% fixos: a exploração é quase nula justamente nas fases longas, onde o agente falha. |
| Replay | ✅ com ressalvas | Taxa de replay = 256/4 = **64 amostras por transição**, contra 8 no DQN de Atari. Funcionou no A/B de 2026-10-17, mas favorece o sobreajuste ao buffer. Os 100 mil passos cobrem só ~40 partidas de ~2 500 passos. O aquecimento é de só 256 transições. |
| γ = 0,95 | ⚠️ Limitante | O horizonte efetivo é de ~20 passos, e o intervalo medido entre comidas é de 25–27 passos (máximo de 262). Uma morte daqui a 60 passos vale −10·0,95⁶⁰ ≈ **−0,46**. |
| Rede 21→256→3 | ✅ suficiente | ~6 mil parâmetros; nem aparece no perfil de CPU. |
| Eficiência | ⚠️ | [CONFIRMADO 2026-11-03] ~3,3 ms por passo e **~95% do tempo em `_reachable_area`** (BFS em Python). São 5 BFS por passo, e uma delas repete a anterior. |

## 4. Pontos fortes a preservar

1. Double DQN, target network, Huber e clip corretos e cobertos por teste.
2. O treino só pelo replay (2026-10-17): +20 pontos sobre o modelo antigo de 30 mil partidas.
3. A seleção do modelo por avaliação gulosa, não pelo recorde.
4. Avaliação determinística, restauração do RNG e comparação pareada com IC.
5. O `load_checkpoint`, que recusa um `STATE_SIZE` diferente em vez de misturar representações.
6. O `train_steps` no checkpoint, que mantém o ε coerente ao retomar.
7. Os 6 testes: [CONFIRMADO 2026-11-03] passam em 10,9 s, em diretório temporário.
8. As features de BFS (`action_analysis`), principal razão de o agente chegar a ~100 pontos.

## 5. Problemas técnicos

### DQN-01 — `best_model.pth` sobrescrito quando o `model.pth` falta · **Crítica**
- **Evidência:**
  - `agent.py:79` inicia `best_eval_mean = -1.0`.
  - Se `load_checkpoint` falha (arquivo ausente, `STATE_SIZE` diferente, pesos inválidos), o valor continua −1.
  - Na partida 500, `agent.py:339` considera "melhor" qualquer média ≥ 0.
- **[CONFIRMADO 2026-11-03]** Num diretório temporário, com uma cópia do modelo de 62 200 partidas como `best_model.pth` e sem `model.pth`: depois de 2 partidas de treino, o `best_model.pth` passou a ter `n_game = 2` e `best_eval_mean = 0`.
- **Quando acontece:** volume novo, `model.pth` apagado ou mudança de `get_state` (o `CLAUDE.md` do projeto prevê mudar o `STATE_SIZE`).
- **Solução:** ler o `best_eval_mean` do próprio `best_model.pth` no início do treino, recusar a sobrescrita quando ele vier de outra execução ou de outro `STATE_SIZE`, ou versionar por execução (DQN-08).
- **Arquivos:** `agent.py`. **Complexidade:** baixa. **Impacto:** protege o artefato mais valioso.

### DQN-02 — Checkpoint truncado derruba o treino; o save não é atômico · **Crítica**
- **[CONFIRMADO 2026-11-03]** Um `model.pth` truncado gera `IndexError: pop from empty list`, que não está na tupla `(OSError, RuntimeError, ValueError)` de `agent.py:110`. O treino cai.
- **[LIDO 2026-11-03]** Com `--restart unless-stopped`, isso vira loop de reinícios. O `torch.save` escreve direto no destino (`agent.py:160`): matar o container durante a gravação produz exatamente esse arquivo.
- **Solução:** gravar em `.tmp` e depois `os.replace`, manter o anterior como `.bak` e, na falha de leitura, parar com mensagem clara em vez de recomeçar em silêncio (o que também evita o DQN-01).
- **Arquivos:** `agent.py`. **Complexidade:** baixa. **Impacto:** alto em treinos longos.

### DQN-03 — Avaliação de seleção ruidosa e enviesada para cima · **Alta**
- **Ruído:** 20 partidas dão IC95% de ±13 a ±14, então diferenças menores que ~15 pontos não são detectáveis.
- **[CONFIRMADO 2026-11-03]** O `best_eval_mean` gravado é **109,15**. O próprio `model.pth` faz **91,8** nas mesmas sementes de seleção (12345+i, 20 partidas) e **101,0** nas sementes 777.
- **Viés:** o valor gravado é o **máximo** de dezenas de estimativas ruidosas sobre as mesmas 20 sementes, então superestima a qualidade real (a "maldição do vencedor").
- **Solução:**
  - avaliar com 50–100 partidas, ou a cada 1 000 partidas;
  - reportar o resultado final em sementes separadas, nunca usadas na seleção;
  - registrar o IC no `eval_log.csv`.
- **Arquivos:** `agent.py`. **Complexidade:** baixa (o custo é CPU).

### DQN-04 — "Mesma sequência de comidas" só vale nas primeiras comidas · **Média**
- **[CONFIRMADO 2026-11-03]** Com a semente 777, `model.pth` e `model10k.pth` divergem já na **7ª comida** (índice 6). O `_place_food` sorteia um índice entre as células livres, e elas dependem do corpo.
- A comparação continua válida (números aleatórios comuns reduzem a variância), mas a frase "remove a sorte da comida" (`agent.py:372`, [[Server DQN]]) é exagerada.
- **Solução:** usar um RNG próprio do ambiente (`random.Random(seed)`) com sorteio independente do corpo (sequência pré-gerada com rejeição), ou apenas corrigir o texto.
- **Arquivos:** `snake_gameai.py`, `agent.py`. **Complexidade:** baixa.

### DQN-05 — O termo de espaço da recompensa é praticamente nulo · **Alta**
- **[CONFIRMADO 2026-11-03]** Em 4 152 passos do `model.pth`: o `|0,05·Δespaço|` tem **mediana 0** e p99 de 0,0006, e passa de 0,01 em só **0,3%** dos passos.
- O progresso vale sempre **±0,10**, porque cada movimento muda a distância Manhattan em exatamente 1.
- A intenção documentada em [[Ambiente Snake]] ("recompensar manter área livre") não se realiza; ver L2.
- **Arquivos:** `snake_gameai.py:127`.

### DQN-06 — Retomar o treino perde o replay buffer · **Média**
- **[LIDO 2026-11-03]** O buffer não é salvo. A cada reinício do container, o treino recomeça com buffer vazio, ε em 0,02 e aquecimento de 256 transições: os primeiros passos de gradiente vêm de poucas partidas, gulosas e correlacionadas.
- A nota [[Docker]] diz que "o treino continua de onde parou"; isso só vale para pesos, contadores e ε.
- **Solução:** aquecimento maior ao retomar (ex.: 20 mil transições) ou salvar o buffer periodicamente (~40 MB em `float32`).
- **Complexidade:** baixa.

### DQN-07 — `play.py` testa o último modelo e duplica o `get_state` · **Alta**
- **[LIDO 2026-11-03]** `play.py:28` carrega `Models/model.pth` (o último), não o melhor. [CONFIRMADO 2026-11-03] O último não é o melhor: média 91,8 contra o `best_eval_mean` de 109,15 nas mesmas sementes.
- `play.py:44-62` copia o `get_state`, e `Test/snake_gameai.py` é outra cópia do ambiente, sem a otimização de BFS. Mudar uma feature exige editar 2–3 lugares sem nenhum teste que perceba.
- **Solução:** módulo único de ambiente e observação, importado pelo treino e pelo `play.py`, com o render do pygame como camada opcional.
- **Complexidade:** média.

### DQN-08 — Experimentos não rastreáveis · **Alta**
- **[LIDO 2026-11-03]** Os hiperparâmetros são constantes de módulo, o checkpoint só registra `train_every_steps` e o Docker roda sem `--seed`.
- **[CONFIRMADO 2026-11-03]** Há 8 `.pth` em `Test/Models` com nomes que não batem com o conteúdo (`model30k.pth` tem 37 000 partidas). Nesta máquina não há `eval_log.csv` nem `best_model.pth`.
- **Solução:**
  - uma `dataclass Config` salva no checkpoint;
  - um diretório por execução: `runs/<data>_<tag>/{config.json, model.pth, best_model.pth, logs/}`;
  - a semente sempre registrada, ainda que aleatória;
  - o hash do commit no checkpoint.
- **Complexidade:** média.

### DQN-09 — Problemas menores · **Baixa**
- **Vitória derruba o treino.** [LIDO 2026-11-03] Com o tabuleiro cheio, `play_step` devolve `food = None` e `train()` chama `get_state(game)` antes do `reset`, o que quebra com `AttributeError` em `game.food.x`. Hoje é inalcançável: o máximo observado foi 166.
- **Timeout.** [LIDO 2026-11-03] O `frame_iteration` é cumulativo por partida (não zera ao comer), não está no estado, e o timeout é tratado como terminal com −10 (truncamento confundido com término). [CONFIRMADO 2026-11-03] Sem efeito prático: **0 de 180** mortes foram por timeout.
- **BFS desperdiçada.** `get_state` faz 3 BFS também no estado terminal, que é descartado, e o `old_space` recalcula o `new_space` do passo anterior.
- **Docker e threads.** O `requirements.txt` instala o `torch` com CUDA, embora `DEVICE` seja CPU, e `set_num_threads(4)` está fixo no código.
- **Log de treino.** O `TrainingLogger` **apaga** o CSV ao passar de 5 GB em vez de rotacionar.

## 6. Limitações algorítmicas

### L1 — Horizonte curto para a fase longa da partida
- **[CONFIRMADO 2026-11-03]** Todas as mortes são por colisão: 36–39 no corpo e 1–4 na parede em cada 40 partidas.
- **[CONFIRMADO 2026-11-03]** Em **80% das partidas** do `model.pth` (32/40), o agente escolheu ao menos uma vez uma ação que leva a uma região menor que o corpo, havendo outra ação com espaço suficiente. Ressalva: é uma heurística, porque a cauda se move.
- **[PROVÁVEL]** Com γ = 0,95, a punição de uma armadilha que se fecha em dezenas de passos quase não chega à decisão que a causou.

### L2 — A recompensa pune o comportamento seguro
- **Aritmética:** afastar-se da comida custa −0,11 por passo, e o termo de espaço é desprezível (DQN-05).
- **[PROVÁVEL]** Um desvio de 20 passos seguindo a cauda custa ≈ −1,4 descontado. Isso é mais que uma morte daqui a 50 passos (−10·0,95⁵⁰ ≈ −0,77): a ganância compensa.
- **[LIDO 2026-11-03]** O comentário "diferenças de potencial" (`snake_gameai.py:125`) é aproximado: shaping baseado em potencial exige `γΦ(s') − Φ(s)`, e o código usa `Φ(s') − Φ(s)`. Com γ < 1, a política ótima não fica formalmente preservada.

### L3 — A representação não expõe a variável decisiva
- **[LIDO 2026-11-03]** O espaço alcançável é normalizado pelo **tabuleiro**, e o comprimento vem separado. Saber se é armadilha exige a comparação *espaço ≥ comprimento*, que a rede precisa aprender sozinha; o L1 mostra que ela não aprendeu bem.
- **[LIDO 2026-11-03]** Direção e dx/dy são **absolutos**, mas as ações são **relativas**: a rede aprende a interação direção × posição quatro vezes. [PROVÁVEL] Um referencial egocêntrico seria mais eficiente em dados.
- **[LIDO 2026-11-03]** O BFS olha **um passo** à frente e trata o corpo como fixo, menos a cauda.
- **Consequência:** o agente é essencialmente uma ponderação aprendida de uma heurística feita à mão. O teto dele é o alcance dessas features.

### L4 — Não existe linha de base
Não há comparação com uma heurística pura, como "ação com mais espaço alcançável, desempate por cauda alcançável e depois por distância à comida". Sem isso, não dá para medir quanto o DQN agrega sobre as próprias features.

## 7. Oportunidades de otimização

| Item | Evidência | Ganho esperado | Complexidade |
|---|---|---|---|
| Cache do `old_space` (reaproveitar o `new_space` anterior) | 5 BFS por passo, 1 repetida | ~15–20% | Baixa |
| Não calcular o `get_state` no terminal | `agent.py:317` | Pequeno | Baixa |
| BFS unificada para as 3 `action_analysis` | 95% do tempo em `_reachable_area` | 2–3× [PROVÁVEL] | Média |
| BFS em NumPy/bitmask ou `scipy.ndimage.label` | idem | 5–10× [PROVÁVEL] | Média |
| Vários ambientes em paralelo | 12 núcleos, treino em 1 processo | Quase linear em amostras/s | Alta |
| GPU | Rede de ~6 mil parâmetros, 1 amostra por passo | Nenhum ou negativo: **não recomendado** | — |
| `torch` CPU no Docker | Imagem com CUDA desnecessária | Imagem bem menor | Baixa |
| Replay em arrays NumPy pré-alocados | `random.sample` indexa a `deque` em O(n) | Pequeno; menos memória | Baixa |

## 8. Experimentos sugeridos

Critério: cada técnica só entra se resolver um problema observado.

1. **Linha de base heurística** · Alta. Responde ao L4 e reaproveita o `action_analysis`.
2. **γ = 0,99 com 3-step returns** · Experimental, alta relevância. Ataca o L1: propaga a punição da morte mais rápido. Reavaliar a frequência da target network e o LR. Testar com o A/B pareado existente, em 3+ sementes.
3. **Revisão da recompensa** · Experimental. Ataca o L2 em três variantes: (a) sem o termo de distância; (b) `γΦ(s') − Φ(s)`; (c) termo de espaço escalado por `espaço < comprimento`.
4. **Features de armadilha e egocêntricas** · Experimental. Ataca o L3 com `espaço_alcançável − comprimento` por ação. Muda o `STATE_SIZE`: corrigir o DQN-01 antes.
5. **Generalização** · Experimental. Rodar o melhor modelo em 20×20 e 40×30 para ver se aprendeu regras ou padrões do 32×24.
6. **Várias sementes de treino** · Alta. Hoje há uma execução longa só. Para afirmar que uma configuração é melhor, são precisas 3+ sementes.
7. **Dueling DQN** · Experimental, baixa relevância. Ajuda quando as ações valem quase o mesmo; não ataca nada observado aqui.
8. **PER** · Experimental. Foi descartado em 2026-10-17 porque o replay quase não era usado ([[Server DQN]]). Hoje o replay é usado, o treino estagnou e as transições de morte na fase longa são raras e informativas. Reavaliar só depois dos itens 2 e 3.
9. **Rede convolucional sobre a grade** · Experimental, longo prazo. É o único caminho além do "BFS de 1 passo", mas custa bem mais amostras e CPU.
10. **Visualização:**
    - um gráfico do `eval_log.csv` com IC;
    - o `play.py` com escolha de checkpoint e semente, para reproduzir uma partida específica;
    - a causa da morte registrada por partida.

## 9. Divergências entre o código e o vault (situação em 2026-11-03)

| Nota | Dizia | Realidade |
|---|---|---|
| [[Server DQN]], "Modelo treinado" | `Test/Models/model.pth` com 14 500 partidas e recorde 101 | [CONFIRMADO] 62 200 partidas, recorde 122, `best_eval_mean` 109,15 e `train_every_steps` 4: um treino longo novo, não registrado |
| [[Server DQN]], "Regressão" | O modelo de 30 000 não estava disponível localmente | `model30k-Antigo.pth` (29 900) e `model35K-Antigo.pth` existem. [CONFIRMADO] O código novo supera o modelo antigo de 30 mil em +19,4 ± 14,5 |
| [[Server DQN]], próximo passo 2 | Treinar do zero até 30 000 | Feito, e passou de 62 mil. O platô a partir de ~37 mil não estava registrado |
| [[Server DQN]] | `model_antigo.pth` tem 6 400 partidas e recorde 90 | `Server_DQN/Models/model_antigo.pth` é um `state_dict` de **11** entradas (v1); `model_antigo02.pth` tem 4 000 partidas e recorde 89. O arquivo de 6 400 não foi encontrado |
| [[Server DQN]] e `agent.py:31` | Todos os modelos enfrentam a mesma sequência de comidas | Só as primeiras ~6 comidas (DQN-04) |
| [[Ambiente Snake]] | `+0,05 × Δespaço` recompensa manter área livre | Efeito numérico nulo (DQN-05) |
| [[Docker]] | O treino continua de onde parou ao recriar o container | Sem o replay buffer (DQN-06); checkpoint corrompido → loop de reinícios (DQN-02) |
| [[Ambiente Snake]] | Anti-loop `frame_iteration > 100·len` | Cumulativo por partida, não por comida (sem efeito prático hoje) |

Conferido e correto: os hiperparâmetros de [[Server DQN]], a fórmula do ε, o Double DQN, o conteúdo do checkpoint, a avaliação a cada 500 partidas com 20 partidas e os 6 testes.

Não encontrados no vault: o registro dos treinos depois de 2026-10-17, as curvas do `eval_log.csv`, experimentos de γ, LR ou recompensa (fora o A/B de 1 000 partidas) e qualquer linha de base.

## 10. Roadmap priorizado

| # | Recomendação | Classe | Arquivos | Complexidade | Impacto |
|---|---|---|---|---|---|
| 1 | Proteger o `best_model.pth` (DQN-01) | **Crítica** | `agent.py` | Baixa | Evita perder o melhor modelo |
| 2 | Save atômico e falha explícita na leitura (DQN-02) | **Crítica** | `agent.py` | Baixa | Evita corrupção e loop de reinícios |
| 3 | Trazer o `best_model.pth` do servidor e compará-lo nas sementes 777 com `model.pth` e `model30k.pth` | **Alta** | — (execução) | Baixa | Saber o nível real hoje |
| 4 | Protocolo de avaliação: mais partidas, sementes separadas para o resultado final, IC no CSV (DQN-03) | **Alta** | `agent.py` | Baixa | Comparações confiáveis |
| 5 | Linha de base heurística (L4) | **Alta** | Script novo | Baixa | Mede o valor real do DQN |
| 6 | Config centralizada, diretório por execução, semente e commit no checkpoint (DQN-08) | **Alta** | `agent.py`, `run_docker.sh` | Média | Reprodutibilidade |
| 7 | Módulo único de ambiente e observação, com o `play.py` usando o `best_model` (DQN-07) | **Alta** | `snake_gameai.py` ×2, `play.py`, `agent.py` | Média | Fim das cópias divergentes |
| 8 | Testes de ambiente e recompensa e um teste do DQN-01 | **Média** | `tests/` | Baixa | Segurança para os experimentos |
| 9 | Otimizações de BFS | **Média** | `snake_gameai.py`, `agent.py` | Baixa–média | 1,3–3× mais amostras por hora |
| 10 | Aquecimento do replay ao retomar (DQN-06) | **Média** | `agent.py` | Baixa | Retomadas mais estáveis |
| 11 | γ = 0,99 com n-step = 3 (L1) | **Experimental** | `agent.py`, `model.py` | Média | Ataca a causa provável do platô |
| 12 | Variantes de recompensa (L2) | **Experimental** | `snake_gameai.py` | Baixa | idem |
| 13 | Features de armadilha e egocêntricas (L3) | **Experimental** | `agent.py`, ambiente | Média | idem; exige o item 1 |
| 14 | Teste de generalização em outros tabuleiros | **Experimental** | Script | Baixa | Responde "generaliza?" |
| 15 | Vários ambientes em paralelo | **Experimental** | Treino | Alta | Escala para os itens 11–13 |
| 16 | PER, Dueling, CNN | **Experimental** | — | Média–alta | Só depois de 11–13 estagnarem |
| 17 | DQN-09 (vitória, timeout, Docker com CPU, log) | **Baixa** | Vários | Baixa | Higiene |

**Trocar de algoritmo hoje é difícil.** Ambiente, observação, política, treino e persistência vivem misturados no `agent.py`. Depois dos itens 6 e 7, uma interface `env.reset()/step()` mais `observe(env)` permitiria comparar DQN, PPO e a linha de base lado a lado.

## Método

- **Diagnóstico por partida:** um script fora do repositório carregou cada checkpoint com `Agent(load_checkpoint=False).load_checkpoint(path)`, jogou partidas gulosas com `random.seed(seed + i)` e classificou a morte (parede, corpo ou timeout).
- **Escolhas que encurralam:** contou as decisões em que a ação escolhida levava a uma região menor que o corpo, havendo outra ação com espaço suficiente.
- **Tempo entre comidas:** mediu o intervalo de passos entre comidas.
- **Perfil e recompensa:** `cProfile` em 3 partidas, e os termos de recompensa recalculados passo a passo.
- **Bug do DQN-01:** reproduzido com `EVAL_EVERY_GAMES = 2` num diretório temporário.
- **Testes:** `python -B -m unittest discover -s tests -v` rodado numa cópia de `DQN/` e `tests/`.
- **Limitação:** o `best_model.pth` e o `eval_log.csv` do servidor não estavam nesta máquina, então o melhor modelo real não foi avaliado.
