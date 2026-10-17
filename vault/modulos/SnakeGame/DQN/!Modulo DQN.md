O módulo **`DQN`** é a **primeira versão do agente por Aprendizado por Reforço**. O agente aprende uma função Q (recompensa futura esperada de cada ação) jogando com a janela do pygame aberta.

Normalmente, um DQN contém:
- **Ambiente**
    - Fornece estado, recompensa e fim de episódio. Ver [[Ambiente Snake]].
- **Rede Q**
    - Aproxima `Q(estado, ação)`. Ver [[Rede e QTrainer]].
- **Replay Buffer**
    - Memória de transições `(s, a, r, s', done)` amostradas aleatoriamente para quebrar a correlação temporal.
- **Política ε-greedy**
    - Equilíbrio entre explorar (ação aleatória) e aproveitar (`argmax Q`).
- **Target Network** (opcional, recomendada)
    - Cópia congelada da rede para calcular o alvo. **Não existe na v1**; foi adicionada na [[Server DQN|v2]].

Arquivos: [[Agente DQN]] (`agent.py` + `Helper.py`) e [[Rede e QTrainer]] (`model.py`).
