import torch
import numpy as np
from snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, DEVICE

INPUT_SIZE = 21
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3

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

game = SnakeGameAI()
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

        if game_over:
            print(f"Game Over! Score: {score}")
            game.reset()
except KeyboardInterrupt:
     print("Teste Encerrado")