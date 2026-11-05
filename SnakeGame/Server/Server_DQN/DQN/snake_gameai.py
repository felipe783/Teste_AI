import random
from collections import namedtuple
from enum import Enum

BLOCK_SIZE = 20

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple("Point", "x y")
# Criado uma vez: _next_point roda milhões de vezes
OFFSETS = {Direction.RIGHT: (BLOCK_SIZE, 0), Direction.LEFT: (-BLOCK_SIZE, 0),
           Direction.UP: (0, -BLOCK_SIZE), Direction.DOWN: (0, BLOCK_SIZE)}
CLOCKWISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

# Vizinhos de cada célula da grade (índice = coluna + linha * colunas), por tamanho de tabuleiro
_NEIGHBOURS = {}


def _neighbours(cols, rows):
    key = (cols, rows)
    if key not in _NEIGHBOURS:
        table = []
        for index in range(cols * rows):
            x, y = index % cols, index // cols
            table.append(tuple(nx + ny * cols for nx, ny in ((x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1))
                               if 0 <= nx < cols and 0 <= ny < rows))
        _NEIGHBOURS[key] = table
    return _NEIGHBOURS[key]


class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.cols, self.rows = w // BLOCK_SIZE, h // BLOCK_SIZE
        self.board_cells = self.cols * self.rows
        self._nbrs = _neighbours(self.cols, self.rows)
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
        self._invalidate()

    def _invalidate(self):
        # As análises dependem só de cabeça, corpo, direção e comida: valem até o próximo movimento
        self._analysis_cache = {}
        self._space = None  # _free_space_ratio do estado atual, se já conhecido

    def _place_food(self):
        occupied = set(self.snake)
        free_cells = [Point(x, y) for x in range(0, self.w, BLOCK_SIZE)
                      for y in range(0, self.h, BLOCK_SIZE)
                      if Point(x, y) not in occupied]
        self.food = random.choice(free_cells) if free_cells else None

    def _inside_board(self, point):
        return 0 <= point.x < self.w and 0 <= point.y < self.h

    def is_collision(self, pt=None):
        pt = self.head if pt is None else pt
        return not self._inside_board(pt) or pt in self.snake[1:]

    @staticmethod
    def _direction_after(direction, action):
        index = CLOCKWISE.index(direction)

        if action[0] == 1:  # [1, 0, 0]: reto
            return CLOCKWISE[index]
        if action[1] == 1:  # [0, 1, 0]: direita
            return CLOCKWISE[(index + 1) % 4]

        return CLOCKWISE[(index - 1) % 4]

    @staticmethod
    def _next_point(point, direction):
        dx, dy = OFFSETS[direction]
        return Point(point.x + dx, point.y + dy)

    def _cell(self, point):
        return point.x // BLOCK_SIZE + (point.y // BLOCK_SIZE) * self.cols

    def _reachable(self, start, blocked):
        """BFS na grade a partir de `start` (dentro do tabuleiro), sem passar por `blocked` (Points).
        Devolve (quantidade de células alcançadas, marcas), em que marcas[célula] == 1 se foi alcançada.
        Laço quente (~95% do tempo do treino): inteiros e bytearray em vez de Point e set."""
        marks = bytearray(self.board_cells)
        cell, inside = self._cell, self._inside_board
        for point in blocked:
            if inside(point):  # num estado terminal a cabeça pode estar fora do tabuleiro
                marks[cell(point)] = 2
        start = cell(start)
        if marks[start]:
            return 0, marks
        marks[start] = 1
        queue, nbrs = [start], self._nbrs
        for current in queue:  # a lista cresce durante o laço: é a fila do BFS
            for neighbour in nbrs[current]:
                if not marks[neighbour]:
                    marks[neighbour] = 1
                    queue.append(neighbour)
        return len(queue), marks

    def action_analysis(self, action):
        """Analisa uma ação no grid: colisão, espaço, rota à comida e à cauda."""
        direction = self._direction_after(self.direction, action)
        cached = self._analysis_cache.get(direction)
        if cached is not None:
            return cached
        next_head = self._next_point(self.head, direction)

        if not self._inside_board(next_head) or next_head in self.snake[1:]:
            result = (True, 0.0, False, False)
        else:
            grows = next_head == self.food
            next_snake = [next_head] + (self.snake if grows else self.snake[:-1])

            # A cauda tende a desocupar, portanto não é uma parede permanente.
            count, marks = self._reachable(next_head, next_snake[1:-1])
            result = (False, count / self.board_cells,
                      marks[self._cell(self.food)] == 1 if self.food is not None else True,
                      marks[self._cell(next_snake[-1])] == 1)
        self._analysis_cache[direction] = result
        return result

    def _free_space_ratio(self):
        if self._space is None:
            count, _ = self._reachable(self.head, self.snake[1:-1])
            self._space = count / self.board_cells
        return self._space

    def play_step(self, action):
        self.frame_iteration += 1
        old_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        old_space = self._free_space_ratio()
        direction = self._direction_after(self.direction, action)
        # Sem comer, o espaço depois do passo é o mesmo BFS que action_analysis já fez para esta ação
        # (cabeça nova, corpo sem as duas últimas células): reaproveitá-lo poupa 1 BFS por passo.
        analysis = self._analysis_cache.get(direction)
        self.direction = direction
        self.head = self._next_point(self.head, self.direction)
        self.snake.insert(0, self.head)
        self._invalidate()

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            return -10.0, True, self.score
        if self.head == self.food:
            self.score += 1
            self._place_food()
            return 10.0, self.food is None, self.score

        self.snake.pop()
        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        self._space = analysis[1] if analysis is not None else None
        new_space = self._free_space_ratio()

        # progress e new_space - old_space são diferenças entre estados, mas sem o fator γ do
        # shaping baseado em potencial (γΦ(s') − Φ(s)); há custo por passo.
        progress = (old_distance - new_distance) / BLOCK_SIZE
        reward = -0.01 + 0.10 * progress + 0.05 * (new_space - old_space)
        return reward, False, self.score
