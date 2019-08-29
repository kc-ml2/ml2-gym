"""modified from YuriyGuts/snake-ai-reinforcement"""
from collections import namedtuple


class Point(namedtuple('PointTuple', ['x', 'y'])):
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)


class Cell:
    EMPTY = 0
    FRUIT = 1
    WALL = 2
    HEAD = [3, 4, 5, 6]
    BODY = [7, 8, 9, 10]

    CELL2SYM = {
        EMPTY: '.',
        FRUIT: 'O',
        WALL: '#',
        **dict(zip(HEAD, ['A', 'B', 'C', 'D'])),
        **dict(zip(BODY, ['a', 'b', 'c', 'd'])),
    }
    SYM2CELL = dict((v, k) for k, v in CELL2SYM.items())


class Direction:
    NORTH = Point(0, -1)
    EAST = Point(1, 0)
    SOUTH = Point(0, 1)
    WEST = Point(-1, 0)
