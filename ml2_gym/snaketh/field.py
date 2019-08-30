"""modified from YuriyGuts/snake-ai-reinforcement"""
import random
import itertools
import numpy as np

from ml2_python.common import Point, Cell, Direction
from ml2_python.python import Python


class Field:
    def __init__(self, init_map=None, init_length=3):
        self.create_cells(init_map)
        # Default layout of 4 players
        self.players = [
            Python(Point(3, 3), Direction.EAST, init_length),
            Python(Point(self.size[1] - 3, 3), Direction.SOUTH, init_length),
            Python(Point(self.size[1] - 3, self.size[0] - 3), Direction.WEST, init_length),
            Python(Point(3, self.size[0] - 3), Direction.NORTH, init_length)
        ]
        _ = self.update_cells()

    def __getitem__(self, point):
        x, y = point
        return self._cells[y, x]

    def __setitem__(self, point, cell):
        x, y = point
        self._cells[y, x] = cell
        if cell == Cell.EMPTY:
            self._empty_cells.add(point)
        elif point in self._empty_cells:
            self._empty_cells.remove(point)

    def __str__(self):
        return '\n'.join(
            ''.join(Cell.CELL2SYM[cell] for cell in row) for row in self._cells
        )

    @property
    def size(self):
        return self._cells.shape

    def get_empty_cell(self):
        return random.choice(list(self._empty_cells))

    def create_cells(self, init_map):
        assert init_map is not None
        self._cells = []
        self._empty_cells = set()

        try:
            for y, line in enumerate(init_map):
                _line = []
                for x, symbol in enumerate(line):
                    cell = Cell.SYM2CELL[symbol]
                    _line.append(cell)
                    if cell == Cell.EMPTY:
                        self._empty_cells.add(Point(x, y))
                self._cells.append(_line)
            self._cells = np.asarray(self._cells)

        except KeyError as err:
            raise ValueError('Unknown map symbol: {}'.format(err.args[0]))

    def update_cells(self):
        conflicts = []
        for idx, python in enumerate(self.players):
            if not python.alive:
                continue

            # Headbutting wall
            if self[python.head] == Cell.WALL:
                conflicts.append((idx,))

            # Headbutting each other
            elif self[python.head] in Cell.HEAD:
                other = Cell.HEAD.index(self[python.head])
                conflicts.append((idx, other))

            # Headbutting body
            elif self[python.head] in Cell.BODY:
                other = Cell.BODY.index(self[python.head])
                conflicts.append((idx, other))

            # At peace
            else:
                self[python.head] = Cell.HEAD[idx]

            for body_cell in itertools.islice(python.body, 1, len(python.body)):
                self[body_cell] = Cell.BODY[idx]

        return conflicts

    def clear(self):
        for python in self.players:
            if not python.alive:
                if self[python.head] in Cell.HEAD:
                    self[python.head] = Cell.EMPTY
                for body_cell in itertools.islice(python.body, 1, len(python.body)):
                    self[body_cell] = Cell.EMPTY
