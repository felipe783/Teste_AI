
[**PyTorch**](https://pytorch.org/)
Framework de Deep Learning usado para criar as redes `Linear_QNet`, calcular gradientes (DQN) e manipular os pesos como tensores (GA). Todos os módulos rodam em `cpu`.
````
pip install torch
````

[**NumPy**](https://numpy.org/)
Representa o vetor de estado do agente (`np.array(..., dtype=np.float32)`) antes da conversão para tensor; também usado no ruído da mutação auto-adaptativa.
````
pip install numpy
````

[**pygame**](https://www.pygame.org/)
Renderiza o jogo Snake em janela nas versões com interface ([[Ambiente Snake]]): DQN v1 e Test.
````
pip install pygame
````

[**OpenCV (opencv-python)**](https://pypi.org/project/opencv-python/)
Grava as partidas em `.mp4`: os frames do pygame são convertidos de RGB para BGR e escritos com `cv2.VideoWriter`.
````
pip install opencv-python
````

[**Matplotlib**](https://matplotlib.org/)
Desenha o gráfico de score e média em tempo real durante os treinos locais (`Helper.py`).
````
pip install matplotlib
````

[**IPython**](https://ipython.org/)
Usado pelo `Helper.py` (`IPython.display`) para limpar e redesenhar o gráfico.
````
pip install ipython
````

**Sudoku**
Usa apenas a biblioteca padrão (`random`, `math`, `json`, `time`, `sys`, `os`).

**Servidor (Docker)**
As imagens instalam somente `torch` e `numpy` (`SnakeGame/Server/*/requirements.txt`), já que rodam sem interface.


Instalar tudo de uma vez:
````
pip install -r SnakeGame/requirements.txt
````

> Em 2026-10-17 a `.venv` da raiz tinha `torch 2.13.0+cpu` e `pygame 2.6.1`, mas não tinha `matplotlib`: os treinos locais falham no `import` do `Helper.py` até instalar.
