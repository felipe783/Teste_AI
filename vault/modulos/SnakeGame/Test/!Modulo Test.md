O módulo **`Test`** é a **bancada de avaliação visual**: carrega um modelo treinado (normalmente vindo do servidor), joga N partidas com janela e guarda em vídeo apenas as partidas recordistas.

Contém:
- [[Play]] (`play.py`): fluxo de teste, vídeos e histórico.
- `snake_gameai.py`: versão com pygame do ambiente do [[Server DQN]] (`SPEED = 40`). Ver [[Ambiente Snake]].
- `model.py`: só a `Linear_QNet`, sem trainer.
- `Models/`, `logs/`, `video/`: dados locais, ignorados pelo Git.
