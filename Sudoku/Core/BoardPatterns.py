from Class.Cell import Cell

easyPattern = [
    [Cell(5,True), Cell(3,True), Cell(), Cell(), Cell(7,True), Cell(), Cell(), Cell(), Cell()],
    [Cell(6,True), Cell(), Cell(), Cell(1,True), Cell(9,True), Cell(5,True), Cell(), Cell(), Cell()],
    [Cell(), Cell(9,True), Cell(8,True), Cell(), Cell(), Cell(), Cell(), Cell(6,True), Cell()],

    [Cell(8,True), Cell(), Cell(), Cell(), Cell(6,True), Cell(), Cell(), Cell(), Cell(3,True)],
    [Cell(4,True),Cell(),Cell(),Cell(8,True),Cell(),Cell(3,True),Cell(),Cell(),Cell(1,True)],
    [Cell(7,True), Cell(), Cell(), Cell(), Cell(2,True), Cell(), Cell(), Cell(), Cell(6,True)],

    [Cell(), Cell(6,True), Cell(), Cell(), Cell(), Cell(), Cell(2,True), Cell(8,True), Cell()],
    [Cell(), Cell(), Cell(), Cell(4,True), Cell(1,True), Cell(9,True), Cell(), Cell(), Cell(5,True)],
    [Cell(), Cell(), Cell(), Cell(), Cell(8,True),Cell(),Cell(),Cell(7,True),Cell(9,True)]
]

mediumPattern = [
    [Cell(8,True), Cell(), Cell(9,True), Cell(2,True), Cell(), Cell(), Cell(), Cell(), Cell(4,True)],
    [Cell(), Cell(), Cell(), Cell(), Cell(), Cell(4,True), Cell(), Cell(), Cell()],
    [Cell(3,True), Cell(), Cell(), Cell(6,True), Cell(), Cell(1,True), Cell(), Cell(), Cell()],

    [Cell(4,True), Cell(9,True), Cell(1,True), Cell(), Cell(6,True), Cell(7,True),Cell(),Cell(),Cell(3,True)],
    [Cell(6,True),Cell(),Cell(),Cell(),Cell(),Cell(9,True),Cell(8,True),Cell(1,True),Cell()],
    [Cell(), Cell(), Cell(), Cell(), Cell(2,True), Cell(3,True), Cell(), Cell(9,True), Cell(6,True  )],

    [Cell(1,True), Cell(), Cell(), Cell(5,True), Cell(), Cell(), Cell(), Cell(), Cell(9,True)],
    [Cell(), Cell(6,True), Cell(4,True), Cell(3,True), Cell(), Cell(2,True), Cell(), Cell(), Cell(5,True)],
    [Cell(), Cell(8,True), Cell(3,True), Cell(9,True), Cell(7,True), Cell(), Cell(), Cell(), Cell()]
]

hardPattern = [
    [Cell(), Cell(), Cell(), Cell(), Cell(5,True), Cell(), Cell(),Cell(),Cell(4,True)],
    [Cell(2,True),Cell(1,True),Cell(3,True),Cell(),Cell(),Cell(4,True),Cell(),Cell(),Cell()],
    [Cell(5,True),Cell(),Cell(),Cell(3,True),Cell(7,True),Cell(),Cell(8,True),Cell(),Cell(6,True)]

    [Cell(), Cell(), Cell(6,True), Cell(), Cell(), Cell(), Cell(7,True), Cell(8,True), Cell()],
    [Cell(), Cell(), Cell(), Cell(1,True), Cell(), Cell(), Cell(6,True), Cell(5,True), Cell()],
    [Cell(), Cell(), Cell(), Cell(), Cell(), Cell(), Cell(), Cell(), Cell(1,True)],

    [Cell(7,True), Cell(8,True), Cell(), Cell(), Cell(1,True),Cell(),Cell(9,True),Cell(),Cell()],
    [Cell(6,True), Cell(),Cell(),Cell(),Cell(),Cell(2,True),Cell(),Cell(),Cell()],
    [Cell(),Cell(2,True),Cell(),Cell(),Cell(9,True),Cell(),Cell(),Cell(),Cell(8,True)]
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
