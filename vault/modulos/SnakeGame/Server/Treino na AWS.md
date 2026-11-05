# Treino na AWS

Como rodar o [[Server DQN]] numa EC2 com os **US$ 100 de créditos**: requisitos medidos, instância recomendada, custos e plano de uso. As correções que tornam o treino retomável estão em [[Correções Snake DQN 2026-11-05]]. Nada foi criado nem executado na AWS: tudo abaixo foi preparado e testado localmente, e os preços precisam ser conferidos antes de usar.

## Requisitos reais (medidos em 2026-11-05)

| Recurso | Medida | Consequência |
|---|---|---|
| CPU | 1 treino = **1 processo de 1 thread a ~99%**; 1 thread é tão rápida quanto 4 | Escala-se rodando **vários experimentos em paralelo**, um por núcleo |
| Vazão | ~2 400 passos/s por treino (Ryzen AI 5 340); ~1 850 cada com 4 em paralelo | ~6–8 mil partidas/hora no regime atual (partidas de treino de ~720 passos + avaliação) [PROVÁVEL]; partidas mais longas (ex.: `--safe-explore`) reduzem esse número |
| RAM | ~275 MB por treino (container, torch de CPU) | 2 GB por vCPU sobram |
| Disco | ~35 MB por execução (modelos, replay e `.bak`) + imagem de 1,4 GB | 20 GB de gp3 bastam |
| GPU | Gradiente = 37% do tempo, ambiente (BFS em Python) = 59% | GPU acelera no máximo 1,6× um treino; uma g4dn.xlarge (~US$ 0,53/h, valor não conferido) custa mais que 3 c7g.xlarge, que rodam 12 treinos: **não compensa** |
| Rede | Nenhuma durante o treino | Transferência desprezível (modelos de 110 KB) |

**Instâncias burstable (t3/t4g) não servem:** o treino usa 100% da CPU sem parar, então elas esgotam os créditos de CPU e caem para a linha de base ou cobram o excedente no modo *unlimited*. [PROVÁVEL]

## Recomendação

**c7g.xlarge (Graviton3, 4 vCPU = 4 núcleos físicos, 8 GB), on-demand, em us-east-1.** Ela roda o lote de 4 experimentos de uma vez.

| Instância | vCPU | Núcleos físicos | On-demand (us-east-1) | Spot (aprox.) | Comentário |
|---|---|---|---|---|---|
| **c7g.xlarge** | 4 | 4 | **US$ 0,145/h** | ~US$ 0,045–0,06/h | Mais barata por núcleo; ARM (o torch de CPU tem arm64) |
| c7a.xlarge | 4 | 4 | ~US$ 0,205/h | ~US$ 0,07/h | Alternativa x86 se o Graviton for lento no benchmark |
| c7i.xlarge | 4 | 2 (*hyperthreading*) | ~US$ 0,1785/h | — | 4 processos disputam 2 núcleos: pior para este caso |
| c7g.2xlarge | 8 | 8 | ~US$ 0,29/h [PROVÁVEL: 2 × xlarge] | — | Para lotes de 8 (replicar com sementes) |

Preços de agregadores (Holori, Vantage, DoIT, Devzero), consultados em 2026-11-05. **Confira no console antes de usar.** Custos extras: IPv4 público a US$ 0,005/h (~US$ 3,65/mês) e EBS gp3 a US$ 0,08/GB-mês (20 GB ≈ US$ 1,60/mês, cobrado mesmo com a instância parada).

**Por que não escolher sem medir:** o desempenho do Python puro (o BFS) por núcleo no Graviton3 em relação ao Ryzen local não foi medido. Por isso o primeiro passo na instância é o `aws/benchmark.sh`: se a c7g ficar abaixo de ~60% da c7a em passos/s, troque para a c7a.xlarge.

**On-demand × spot:** o treino aguenta interrupção (SIGTERM salva tudo, há checkpoint a cada 10 min e a retomada é automática), então a spot economiza ~60%. Mas o lote inteiro custa poucos dólares em on-demand, e uma spot com "terminate" apaga o disco raiz se ele tiver `DeleteOnTermination`. Comece com on-demand; a spot só compensa para lotes longos, com volume que sobreviva ou com interrupção do tipo *stop*.

## Estimativa de custos

| Item | Conta | Custo |
|---|---|---|
| Lote de 4 experimentos × 40 000 partidas | ~6–10 h de c7g.xlarge [PROVÁVEL] | ~US$ 1–1,50 |
| Replicação: 6–8 execuções × 40 000 partidas (sementes 2 e 3) | ~10 h de c7g.2xlarge | ~US$ 3 |
| Disco e IPv4 durante um mês | 20 GB gp3 + IP | ~US$ 5,30 |
| **Erro a evitar: esquecer a c7g.xlarge ligada** | 24 h × 30 dias | **~US$ 108/mês: estoura o crédito** |

## Estratégia para os US$ 100

1. **Proteções antes de lançar:** crie um *AWS Budget* com alertas em US$ 10, 25 e 50 por e-mail. Confira em *Billing → Credits* a validade e os serviços elegíveis dos créditos.
2. **Fase 0 — preparo (< US$ 0,20):** suba a c7g.xlarge com Amazon Linux 2023 e 20 GB de gp3. Rode `bash aws/setup_ec2.sh`, entre de novo no SSH e rode `bash aws/benchmark.sh`. Compare o resultado com a referência impressa.
3. **Fase 1 — lote 1 (~US$ 1,50):** `AUTO_SHUTDOWN=1 bash aws/experiments.sh`. São 4 execuções do zero, com 40 000 partidas e semente 1: `base`, `safe`, `g099n3` e `safe_g099n3`. A instância desliga sozinha ao terminar. Acompanhe com `bash aws/status.sh`.
4. **Avaliação final, fora da seleção:** traga os resultados (`rsync -av ec2-user@<ip>:runs/ runs_aws/`) e compare cada `best_model.pth` com o do `base`, de forma pareada: `python DQN/agent.py --eval --episodes 100 --seed 777 --model runs_aws/base_s1/Models/best_model.pth runs_aws/<exp>/Models/best_model.pth`.
5. **Fase 2 — replicar (~US$ 3–5):** a melhor configuração e o `base`, com as sementes 2 e 3. Uma semente só não basta para afirmar que uma configuração é melhor.
6. **Fase 3 — escalar o que funcionou (~US$ 10–20):** execuções longas (100 mil partidas ou mais) da melhor configuração e os próximos experimentos da auditoria (recompensa e features de armadilha, que exigem código novo).
7. **Reserva:** mantenha uns US$ 50 sem uso planejado, para erros e para a instância ficar parada com disco. **Termine** a instância (não só pare) e apague os volumes quando acabar.

Com essa vazão, US$ 100 compram ~650 h de c7g.xlarge, ou seja, ~2 600 horas de treino. O limite real é **ter experimentos bem desenhados**, não o crédito.

## Execução desacoplada do terminal

- **Docker (recomendado):** o `run_docker.sh` e o `experiments.sh` criam containers que sobrevivem ao fim do SSH. `RESTART_POLICY=on-failure:5` com `--games` evita reiniciar o que já terminou; `unless-stopped` serve para treinos sem fim.
- **Sem Docker:** `nohup python DQN/agent.py --run-dir runs/x --games 40000 > runs/x.out 2>&1 &`, ou uma sessão de `tmux`.
- **Parar com segurança:** `docker stop -t 120 <nome>` (envia SIGTERM; o treino salva e sai). Desligar a instância pelo console também manda SIGTERM aos containers.
- **Retomar:** rode o mesmo comando. O `config.json` da pasta guarda os hiperparâmetros, e só o que você passar na linha de comando muda (ex.: `--games` maior).

## Riscos

- Instância esquecida ligada (ver a tabela de custos); o `AUTO_SHUTDOWN=1` e o Budget mitigam.
- A imagem arm64 não foi construída localmente (só x86_64); o primeiro `docker build` na c7g é o teste.
- Estimativas de horas marcadas [PROVÁVEL]: partidas mais longas, que são justamente o objetivo dos experimentos, deixam o treino mais lento.
