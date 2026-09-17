import random
from collections import deque, namedtuple
from enum import Enum

BLOCK_SIZE = 20

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple("Point", "x y")

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        free_cells = [Point(x, y) for x in range(0, self.w, BLOCK_SIZE)
                      for y in range(0, self.h, BLOCK_SIZE)
                      if Point(x, y) not in self.snake]
        self.food = random.choice(free_cells) if free_cells else None

    def _inside_board(self, point):
        return 0 <= point.x < self.w and 0 <= point.y < self.h

    def is_collision(self, pt=None):
        pt = self.head if pt is None else pt
        return not self._inside_board(pt) or pt in self.snake[1:]

    @staticmethod
    def _direction_after(direction, action):
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clockwise.index(direction)

        if action == [1, 0, 0]:
            return clockwise[index]
        if action == [0, 1, 0]:
            return clockwise[(index + 1) % 4]
        
        return clockwise[(index - 1) % 4]

    @staticmethod
    def _next_point(point, direction):
        offsets = {Direction.RIGHT: (BLOCK_SIZE, 0), Direction.LEFT: (-BLOCK_SIZE, 0),
                   Direction.UP: (0, -BLOCK_SIZE), Direction.DOWN: (0, BLOCK_SIZE)}
        dx, dy = offsets[direction]
        return Point(point.x + dx, point.y + dy)

    def _reachable_area(self, start, blocked):
        if not self._inside_board(start) or start in blocked:
            return set()
        visited, queue = {start}, deque([start])

        while queue:
            point = queue.popleft()

            for direction in Direction:
                neighbour = self._next_point(point, direction)

                if (self._inside_board(neighbour) and neighbour not in blocked
                        and neighbour not in visited):
                    visited.add(neighbour)
                    queue.append(neighbour)
        return visited

    def action_analysis(self, action):
        """Analisa uma ação no grid: colisão, espaço, rota à comida e à cauda."""
        direction = self._direction_after(self.direction, action)
        next_head = self._next_point(self.head, direction)

        if not self._inside_board(next_head) or next_head in self.snake[1:]:
            return True, 0.0, False, False

        grows = next_head == self.food
        next_snake = [next_head] + (self.snake if grows else self.snake[:-1])

        # A cauda tende a desocupar, portanto não é uma parede permanente.
        reachable = self._reachable_area(next_head, set(next_snake[1:-1]))
        board_cells = (self.w // BLOCK_SIZE) * (self.h // BLOCK_SIZE)
        return (False, len(reachable) / board_cells,
                self.food in reachable if self.food is not None else True,
                next_snake[-1] in reachable)

    def _free_space_ratio(self):
        board_cells = (self.w // BLOCK_SIZE) * (self.h // BLOCK_SIZE)
        return len(self._reachable_area(self.head, set(self.snake[1:-1]))) / board_cells

    def play_step(self, action):
        self.frame_iteration += 1
        old_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        old_space = self._free_space_ratio()
        self.direction = self._direction_after(self.direction, action)
        self.head = self._next_point(self.head, self.direction)
        self.snake.insert(0, self.head)

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            return -10.0, True, self.score
        if self.head == self.food:
            self.score += 1
            self._place_food()
            return 10.0, self.food is None, self.score

        self.snake.pop()
        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        new_space = self._free_space_ratio()
        
        # Diferenças de potencial: dar voltas não acumula bônus; há custo por passo.
        progress = (old_distance - new_distance) / BLOCK_SIZE
        reward = -0.01 + 0.10 * progress + 0.05 * (new_space - old_space)
        return reward, False, self.score
