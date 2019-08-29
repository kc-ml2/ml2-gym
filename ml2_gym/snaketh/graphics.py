"""modified from YuriyGuts/snake-ai-reinforcement"""
import numpy as np
import torch
import pygame
from ml2_python.common import Cell
from ml2_python.environment import Action
from runner import reshape_s


class Color:
    BACKGROUND = (170, 204, 153)
    CELL = {
        Cell.EMPTY: BACKGROUND,
        Cell.FRUIT: (173, 52, 80),
        Cell.WALL: (56, 56, 56),

        Cell.HEAD[0]: (105, 132, 164),
        Cell.BODY[0]: (105, 132, 164),

        Cell.HEAD[1]: (105, 132, 164),
        Cell.BODY[1]: (105, 132, 164),

        Cell.HEAD[2]: (105, 132, 164),
        Cell.BODY[2]: (105, 132, 164),

        Cell.HEAD[3]: (105, 132, 164),
        Cell.BODY[3]: (105, 132, 164),
    }


class Timer:
    FPS_LIMIT = 60

    def __init__(self):
        self.reset()

    def reset(self):
        self.fps_clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()

    def tick(self):
        self.fps_clock.tick(self.FPS_LIMIT)

    @property
    def time(self):
        return pygame.time.get_ticks() - self.start_time


class ML2PythonGUI:
    def __init__(self, env, args):
        pygame.init()
        self.env = env
        self.human = args.human
        self.cell_size = args.cell_size
        self.device = args.device

        self.screen = pygame.display.set_mode((
            self.env.field.size[0]*self.cell_size,
            self.env.field.size[1]*self.cell_size
        ))
        self.screen.fill(Color.BACKGROUND)
        pygame.display.set_caption('ML2 Python')

        self.timer = Timer()
        self.reset()

    def reset(self):
        self.timer.reset()
        return self.env.reset()

    def run(self, policy):
        done = False
        obs = self.reset()

        while not done:
            self.render()
            pygame.display.update()

            action = Action.IDLE
            for event in pygame.event.get():
                if self.human and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = Action.TURN_LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = Action.TURN_RIGHT
                    elif event.key == pygame.K_ESCAPE:
                        raise
                elif event.type == pygame.QUIT:
                    break

            timed_out = self.timer.time >= self.timer.FPS_LIMIT
            made_move = self.human and action != Action.IDLE
            if timed_out or made_move:
                self.timer.reset()

                actions = []
                for idx in range(self.env.num_players):
                    if not (idx == 0 and self.human):
                        state = reshape_s(obs, idx, n=self.env.num_players)
                        state = torch.tensor(state).to(self.device)
                        q = policy(state.unsqueeze(0).float())
                        action = torch.argmax(q, dim=1).item()
                    actions.append(action)
                actions = np.asarray(actions)

                obs, _, dones, _ = self.env.step(actions)
                done = all(dones)
                self.timer.tick()

    def render(self):
        for x in range(self.env.field.size[0]):
            for y in range(self.env.field.size[1]):
                coords = pygame.Rect(
                    x*self.cell_size,
                    y*self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                color = Color.CELL[self.env.field[x, y]]
                pygame.draw.rect(self.screen, color, coords, 1)

                padding = self.cell_size // 6 * 2
                coords = coords.inflate((-padding, padding))
                pygame.draw.rect(self.screen, color, coords)
