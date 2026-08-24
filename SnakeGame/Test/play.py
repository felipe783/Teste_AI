import torch
import numpy as np
from snake_gameai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, DEVICE

MODEL_PATH = "Models/best_genetic.pth"

INPUT_SIZE = 15
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3

def get_state(game):
        head = game.snake[0]

        point_l = Point(head.x - BLOCK_SIZE,head.y)
        point_r = Point(head.x + BLOCK_SIZE,head.y)
        point_u = Point(head.x,head.y - BLOCK_SIZE)
        point_d = Point(head.x,head.y + BLOCK_SIZE)

        # Ver 2 Blocos pra frente
        point_l2 = Point(head.x - (2 * BLOCK_SIZE),head.y)
        point_r2 = Point(head.x + (2 * BLOCK_SIZE),head.y)
        point_u2 = Point(head.x,head.y - (2 * BLOCK_SIZE))
        point_d2 = Point(head.x,head.y + (2 * BLOCK_SIZE))

        dir_l = (game.direction == Direction.LEFT)
        dir_r = (game.direction == Direction.RIGHT)
        dir_u = (game.direction == Direction.UP)
        dir_d = (game.direction == Direction.DOWN)

        state = [
            # Perigo a frente
            (
                (dir_u and game.is_collision(point_u)) or
                (dir_d and game.is_collision(point_d)) or
                (dir_l and game.is_collision(point_l)) or
                (dir_r and game.is_collision(point_r))
            ),

            # Perigo a direita
            (
                (dir_u and game.is_collision(point_r)) or
                (dir_d and game.is_collision(point_l)) or
                (dir_l and game.is_collision(point_u)) or
                (dir_r and game.is_collision(point_d))
            ),

            # Perigo a Esquerda
            (
                (dir_u and game.is_collision(point_l)) or
                (dir_d and game.is_collision(point_r)) or
                (dir_l and game.is_collision(point_d)) or
                (dir_r and game.is_collision(point_u))
            ),

            # Direcao
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Comida
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,

            # Perigo 2 Blocos
            (
                (dir_u and game.is_collision(point_u2)) or
                (dir_d and game.is_collision(point_d2)) or
                (dir_l and game.is_collision(point_l2)) or
                (dir_r and game.is_collision(point_r2))
            ),

            # Perigo a direita
            (
                (dir_u and game.is_collision(point_r2)) or
                (dir_d and game.is_collision(point_l2)) or
                (dir_l and game.is_collision(point_u2)) or
                (dir_r and game.is_collision(point_d2))
            ),

            # Perigo a esquerda
            (
                (dir_u and game.is_collision(point_l2)) or
                (dir_d and game.is_collision(point_r2)) or
                (dir_l and game.is_collision(point_d2)) or
                (dir_r and game.is_collision(point_u2))
            ),

            # Tamanho da Cobra
            len(game.snake) / ((game.w // BLOCK_SIZE) * (game.h // BLOCK_SIZE))
        ]
        return np.array(state,dtype=np.float32)



model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

checkpoint = torch.load(MODEL_PATH,map_location=DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

game = SnakeGameAI()

while True:
    state = get_state(game)
    state_tensor = torch.tensor(state,dtype=torch.float32)

    with torch.no_grad():
        prediction = model(state_tensor)

    action_index = torch.argmax(prediction).item()

    action = np.zeros(3)
    action[action_index] = 1

    reward, game_over, score = game.play_step(action)

    if game_over:
        print(f"Game Over! Score: {score}")
        game.reset()