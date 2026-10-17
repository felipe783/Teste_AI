O módulo **`Server`** contém as versões do treino preparadas para rodar **sem interface, dentro de containers Docker**, num servidor Linux (`/home/felipe/Teste_IA/...`).

Normalmente, essa camada contém:
- **Código headless**
    - Sem pygame/OpenCV/Matplotlib, só `torch` e `numpy`.
- **Imagem Docker**
    - `Dockerfile` com a aplicação e as dependências.
- **Script de implantação**
    - `run_docker.sh`: build, recriação do container e montagem de volumes.
- **Persistência fora do container**
    - Modelos e logs em volumes do host, para sobreviver a `docker rm`.

No Teste_AI:
- [[Server DQN]]: **DQN v2**, a versão mais avançada do agente. Funciona.
- [[Server GA]]: cópia adaptada do GA. **Não funciona** no estado atual.
- [[Docker]]: `Dockerfile` e `run_docker.sh` das duas.
