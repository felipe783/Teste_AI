import random
from enum import Enum
from collections import namedtuple


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x y')

BLOCK_SIZE = 20


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(
            self.w // 2,
            self.h // 2
        )

        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]

        self.score = 0
        self.frame_iteration = 0

        self._place_food()

    def _place_food(self):
        x = random.randint(0,(self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0,(self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):

        self.frame_iteration += 1

        # Move
        self._move(action)
        self.snake.insert(0, self.head)

        # Collision
        reward = 0
        game_over = False

        if (self.is_collision()or self.frame_iteration > 100 * len(self.snake)):
            game_over = True
            reward = -10

            return reward, game_over, self.score

        # Food
        if self.head == self.food:
            self.score += 1
            reward = 10

            self._place_food()

        else:
            self.snake.pop()

        return reward, game_over, self.score

    def _move(self, action):

        clock_wise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP
        ]

        idx = clock_wise.index(self.direction)

        if action == [1, 0, 0]:
            new_dir = clock_wise[idx]

        elif action == [0, 1, 0]:
            new_dir = clock_wise[(idx + 1) % 4]

        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE

        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE

        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def is_collision(self, pt=None):

        if pt is None:
            pt = self.head

        # Parede
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True

        # Corpo
        if pt in self.snake[1:]:
            return True

        return False