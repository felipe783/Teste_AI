from datetime import datetime
from pathlib import Path

import cv2
import pygame
import torch
import numpy as np
from snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, DEVICE

INPUT_SIZE = 21
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3
VIDEO_DIR = Path(__file__).resolve().parent / "video"
VIDEO_FPS = 40

print("------------\n")
print("1 -- Modelo treinado por Recompensa")
print("2 -- Modelo treinado por Geracoes\n")
print("------------")
escolha = int(input("Numero: "))

if escolha == 1:
    MODEL_PATH = "Models/model.pth"
elif escolha == 2:
    MODEL_PATH = "Models/best_genetic.pth"
else:
    print("Escolhe certo")
    exit()

def get_state(game):
        head, tail = game.head, game.snake[-1]
        actions = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        analyses = [game.action_analysis(action) for action in actions]
        board_width, board_height = max(1, game.w - BLOCK_SIZE), max(1, game.h - BLOCK_SIZE)
        capacity = (game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE)

        state = [
            *(game.direction == direction for direction in
              (Direction.LEFT, Direction.RIGHT, Direction.UP, Direction.DOWN)),
            (game.food.x - head.x) / board_width, (game.food.y - head.y) / board_height,
            (tail.x - head.x) / board_width, (tail.y - head.y) / board_height,
            *(item[0] for item in analyses),       # colisão: frente, direita, esquerda
            *(item[1] for item in analyses),       # fração de espaço acessível
            *(item[2] for item in analyses),       # rota segura até a comida
            *(item[3] for item in analyses),       # rota até a cauda/ciclo de escape
            len(game.snake) / capacity,
        ]
        return np.asarray(state, dtype=np.float32)



model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

checkpoint = torch.load(MODEL_PATH,map_location=DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


def criar_gravador(game):
    """Cria um arquivo MP4 novo para a partida atual."""
    VIDEO_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    caminho = VIDEO_DIR / f"snake_{timestamp}.mp4"
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    gravador = cv2.VideoWriter(str(caminho), codec, VIDEO_FPS, (game.w, game.h))

    if not gravador.isOpened():
        raise RuntimeError(f"Nao foi possivel criar o video: {caminho}")

    print(f"Gravando partida em: {caminho}")
    return gravador, caminho


def finalizar_gravacao(gravador, caminho, score):
    """Fecha o video e acrescenta a pontuacao final ao nome do arquivo."""
    gravador.release()
    caminho_final = caminho.with_name(f"{caminho.stem}_score_{score}.mp4")
    caminho.rename(caminho_final)
    print(f"Video salvo em: {caminho_final}")


def gravar_frame(gravador, game):
    """Copia o frame mostrado pelo Pygame para o arquivo de video."""
    frame_rgb = pygame.surfarray.array3d(game.display)
    frame_bgr = cv2.cvtColor(np.transpose(frame_rgb, (1, 0, 2)), cv2.COLOR_RGB2BGR)
    gravador.write(frame_bgr)


game = SnakeGameAI()
gravador, caminho_video = criar_gravador(game)

try:
    while True:
        state = get_state(game)
        state_tensor = torch.tensor(state,dtype=torch.float32)

        with torch.no_grad():
            prediction = model(state_tensor)

        action_index = torch.argmax(prediction).item()

        action = [0, 0, 0]
        action[action_index] = 1

        reward, game_over, score = game.play_step(action)
        gravar_frame(gravador, game)

        if game_over:
            print(f"Game Over! Score: {score}")
            finalizar_gravacao(gravador, caminho_video, score)
            game.reset()
            gravador, caminho_video = criar_gravador(game)
except KeyboardInterrupt:
    print("Teste Encerrado")
finally:
    finalizar_gravacao(gravador, caminho_video, game.score)
    pygame.quit()
