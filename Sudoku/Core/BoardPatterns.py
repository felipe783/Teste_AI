from Class.Cell import Cell

easyPattern = [
    [Cell(5), Cell(3), Cell(), Cell(), Cell(7), Cell(), Cell(), Cell(), Cell()],
    [Cell(6), Cell(), Cell(), Cell(1), Cell(9), Cell(5), Cell(), Cell(), Cell()],
    [Cell(), Cell(9), Cell(8), Cell(), Cell(), Cell(), Cell(), Cell(6), Cell()],

    [Cell(8), Cell(), Cell(), Cell(), Cell(6), Cell(), Cell(), Cell(), Cell(3)],
    [Cell(4), Cell(), Cell(), Cell(8), Cell(), Cell(3), Cell(), Cell(), Cell(1)],
    [Cell(7), Cell(), Cell(), Cell(), Cell(2), Cell(), Cell(), Cell(), Cell(6)],

    [Cell(), Cell(6), Cell(), Cell(), Cell(), Cell(), Cell(2), Cell(8), Cell()],
    [Cell(), Cell(), Cell(), Cell(4), Cell(1), Cell(9), Cell(), Cell(), Cell(5)],
    [Cell(), Cell(), Cell(), Cell(), Cell(8), Cell(), Cell(), Cell(7), Cell(9)]
]

mediumPattern = [
    [Cell(8), Cell(), Cell(9), Cell(2), Cell(), Cell(), Cell(), Cell(), Cell(4)],
    [Cell(), Cell(), Cell(), Cell(), Cell(), Cell(4), Cell(), Cell(), Cell()],
    [Cell(3), Cell(), Cell(), Cell(6), Cell(), Cell(1), Cell(), Cell(), Cell()],

    [Cell(4), Cell(9), Cell(1), Cell(), Cell(6), Cell(7), Cell(), Cell(), Cell(3)],
    [Cell(6), Cell(), Cell(), Cell(), Cell(), Cell(9), Cell(8), Cell(1), Cell()],
    [Cell(), Cell(), Cell(), Cell(), Cell(2), Cell(3), Cell(), Cell(9), Cell(6)],

    [Cell(1), Cell(), Cell(), Cell(5), Cell(), Cell(), Cell(), Cell(), Cell(9)],
    [Cell(), Cell(6), Cell(4), Cell(3), Cell(), Cell(2), Cell(), Cell(), Cell(5)],
    [Cell(), Cell(8), Cell(3), Cell(9), Cell(7), Cell(), Cell(), Cell(), Cell()]
]

hardPattern = [
    [Cell(), Cell(), Cell(), Cell(), Cell(5), Cell(), Cell(), Cell(), Cell(4)],
    [Cell(2), Cell(1), Cell(3), Cell(), Cell(), Cell(4), Cell(), Cell(), Cell()],
    [Cell(5), Cell(), Cell(), Cell(3), Cell(7), Cell(), Cell(8), Cell(), Cell(6)],

    [Cell(), Cell(), Cell(6), Cell(), Cell(), Cell(), Cell(7), Cell(8), Cell()],
    [Cell(), Cell(), Cell(), Cell(1), Cell(), Cell(), Cell(6), Cell(5), Cell()],
    [Cell(), Cell(), Cell(), Cell(), Cell(), Cell(), Cell(), Cell(), Cell(1)],

    [Cell(7), Cell(8), Cell(), Cell(), Cell(1), Cell(), Cell(9), Cell(), Cell()],
    [Cell(6), Cell(), Cell(), Cell(), Cell(), Cell(2), Cell(), Cell(), Cell()],
    [Cell(), Cell(2), Cell(), Cell(), Cell(9), Cell(), Cell(), Cell(), Cell(8)]
]

victoriousBoard = [
    [Cell(7), Cell(1), Cell(5), Cell(3), Cell(4), Cell(9), Cell(8), Cell(2), Cell(6)],
    [Cell(4), Cell(2), Cell(6), Cell(1), Cell(8), Cell(7), Cell(3), Cell(9), Cell(5)],
    [Cell(3), Cell(8), Cell(9), Cell(5), Cell(2), Cell(6), Cell(7), Cell(4), Cell(1)],

    [Cell(1), Cell(7), Cell(2), Cell(6), Cell(9), Cell(4), Cell(5), Cell(3), Cell(8)],
    [Cell(8), Cell(6), Cell(4), Cell(7), Cell(3), Cell(5), Cell(9), Cell(1), Cell(2)],
    [Cell(5), Cell(9), Cell(3), Cell(2), Cell(1), Cell(8), Cell(4), Cell(6), Cell(7)],

    [Cell(6), Cell(4), Cell(7), Cell(9), Cell(5), Cell(2), Cell(1), Cell(8), Cell(3)],
    [Cell(9), Cell(5), Cell(1), Cell(8), Cell(6), Cell(3), Cell(2), Cell(7), Cell(4)],
    [Cell(2), Cell(3), Cell(8), Cell(4), Cell(7), Cell(1), Cell(6), Cell(5), Cell(9)]
]

boardPatterns = [
    easyPattern,
    mediumPattern,
    hardPattern,
    victoriousBoard
]
