VALUES = 0
FIXED_MASKS = 1

def _make_board(values):
    """Monta os arrays de valores e de células fixas a partir de zeros."""
    fixed_masks = [
        sum(1 << column for column, value in enumerate(row) if value != 0)
        for row in values
    ]
    return [values, fixed_masks]


easyPattern = _make_board([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
])

mediumPattern = _make_board([
    [8, 0, 9, 2, 0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 4, 0, 0, 0],
    [3, 0, 0, 6, 0, 1, 0, 0, 0],
    [4, 9, 1, 0, 6, 7, 0, 0, 3],
    [6, 0, 0, 0, 0, 9, 8, 1, 0],
    [0, 0, 0, 0, 2, 3, 0, 9, 6],
    [1, 0, 0, 5, 0, 0, 0, 0, 9],
    [0, 6, 4, 3, 0, 2, 0, 0, 5],
    [0, 8, 3, 9, 7, 0, 0, 0, 0],
])

hardPattern = _make_board([
    [0, 0, 0, 0, 5, 0, 0, 0, 4],
    [2, 1, 3, 0, 0, 4, 0, 0, 0],
    [5, 0, 0, 3, 7, 0, 8, 0, 6],
    [0, 0, 6, 0, 0, 0, 7, 8, 0],
    [0, 0, 0, 1, 0, 0, 6, 5, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [7, 8, 0, 0, 1, 0, 9, 0, 0],
    [6, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 0, 9, 0, 0, 0, 8],
])

# Este tabuleiro é a resposta completa, não um enunciado: nenhuma posição é fixa.
victoriousBoard = [
    [
        [7, 1, 5, 3, 4, 9, 8, 2, 6],
        [4, 2, 6, 1, 8, 7, 3, 9, 5],
        [3, 8, 9, 5, 2, 6, 7, 4, 1],
        [1, 7, 2, 6, 9, 4, 5, 3, 8],
        [8, 6, 4, 7, 3, 5, 9, 1, 2],
        [5, 9, 3, 2, 1, 8, 4, 6, 7],
        [6, 4, 7, 9, 5, 2, 1, 8, 3],
        [9, 5, 1, 8, 6, 3, 2, 7, 4],
        [2, 3, 8, 4, 7, 1, 6, 5, 9],
    ],
    [0] * 9,
]

boardPatterns = [
    easyPattern,
    mediumPattern,
    hardPattern,
    victoriousBoard,
]
