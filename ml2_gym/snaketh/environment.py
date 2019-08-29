import numpy as np
import gym
from gym.spaces import Discrete, Box
from ml2_python.common import Cell
from ml2_python.field import Field


class Action:
    IDLE = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


class Reward:
    FRUIT = 10
    KILL = 50
    LOSE = -100
    WIN = 500
    # count-based bonus
    alpha = 0.01
    beta = 0.05


class ML2Python(gym.Env):
    def __init__(self, init_map, init_length=3, fruit_interval=10):
        self.init_map = init_map
        self.init_length = init_length
        self.fruit_interval = fruit_interval

        self.field = Field(self.init_map, self.init_length)
        self.num_players = len(self.field.players)
        self.observation_space = Box(
            low=0,
            high=2,
            shape=(self.num_players + 2, *self.field.size)
        )
        self.action_space = Discrete(3)

        self.reset()

    def reset(self):
        self.field = Field(self.init_map, self.init_length)
        self.visits = np.zeros((self.num_players, np.prod(self.field.size)))
        self.dones = np.zeros(self.num_players, dtype=bool)
        self.epinfos = {
            'step': 0,
            'scores': np.zeros(self.num_players),
            'fruits': np.zeros(self.num_players),
            'kills': np.zeros(self.num_players)
        }
        return self.encode()

    def step(self, actions):
        assert len(actions) == self.num_players
        rewards = np.zeros(self.num_players)
        for idx, action in enumerate(actions):
            python = self.field.players[idx]
            if not python.alive:
                continue

            # Choose action
            if action == Action.TURN_LEFT:
                python.turn_left()
            elif action == Action.TURN_RIGHT:
                python.turn_right()

            # Eat fruit
            if self.field[python.next] == Cell.FRUIT:
                python.grow()
                rewards[idx] += Reward.FRUIT
                self.epinfos['fruits'][idx] += 1

            # Or just starve
            else:
                self.field[python.tail] = Cell.EMPTY
                python.move()

            self.field.players[idx] = python

            # Add count-based bonus
            cell = int(python.head.x + python.head.y*self.field.size[0])
            self.visits[idx][cell] += 1
            rewards[idx] += Reward.beta*(self.visits[idx][cell] + Reward.alpha)**(-0.5)

        # Resolve conflicts
        conflicts = self.field.update_cells()
        for conflict in conflicts:
            idx = conflict[0]
            python = self.field.players[idx]
            python.alive = False
            self.dones[idx] = True
            rewards[idx] += Reward.LOSE

            # If collided with another player
            if len(conflict) > 1:
                idx = conflict[1]
                if idx != conflict[0]:
                    other = self.field.players[idx]
                    # Head to head
                    if self.field[python.head] in Cell.HEAD:
                        other.alive = False
                        self.dones[idx] = True
                        rewards[idx] += Reward.LOSE
                    # Head to body
                    else:
                        rewards[idx] += Reward.KILL
                        self.epinfos['kills'][idx] += 1

        # Check if done and calculate scores
        if np.sum(~self.dones) == 1:
            idx = list(self.dones).index(False)
            self.dones[idx] = True
            rewards[idx] += Reward.WIN
        self.epinfos['scores'] += rewards

        # Generate fruits
        self.epinfos['step'] += 1
        if self.epinfos['step'] % self.fruit_interval == 0:
            pos = self.field.get_empty_cell()
            self.field[pos] = Cell.FRUIT

        return self.encode(), rewards, self.dones, self.epinfos

    def encode(self):
        self.field.clear()
        state = np.zeros(self.observation_space.shape)
        for idx in range(self.num_players):
            head = np.isin(self.field._cells,
                           Cell.HEAD[idx]).astype(np.float32)
            body = np.isin(self.field._cells,
                           Cell.BODY[idx]).astype(np.float32)
            state[idx] = 2*head + body

        state[-2] = np.isin(self.field._cells, Cell.FRUIT).astype(np.float32)
        state[-1] = np.isin(self.field._cells, Cell.WALL).astype(np.float32)
        return state

    def render(self):
        print(self.field)
