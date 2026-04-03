import numpy as np
from .spread import compute_spread_prob
from .ember import ember_step
from visualization.animation import animate_fire

UNBURNED, BURNING, BURNED = 0, 1, 2

class WildfireModel:

    def __init__(self, grid_size=100):
        self.grid_size = grid_size

    def initialize(self):
        n = self.grid_size

        self.state = np.zeros((n, n), dtype=int)
        self.fuel = np.random.uniform(0.5, 1.0, (n, n))
        self.moisture = np.random.uniform(0.2, 0.6, (n, n))

        self.structures = (np.random.rand(n, n) < 0.1)
        self.hardening = np.zeros((n, n))
        self.hardening[self.structures] = np.random.uniform(0.3, 0.8, np.sum(self.structures))

        self.state[n//2, n//2] = BURNING

    def neighbors(self, i, j):
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                    yield ni, nj

    def step(self):
        new_state = self.state.copy()

        for i in range(self.grid_size):
            for j in range(self.grid_size):

                if self.state[i, j] == BURNING:
                    for ni, nj in self.neighbors(i, j):
                        if self.state[ni, nj] == UNBURNED:

                            p = compute_spread_prob(
                                i, j, ni, nj,
                                self.fuel, self.moisture,
                                self.structures, self.hardening
                            )

                            if np.random.rand() < p:
                                new_state[ni, nj] = BURNING

                    new_state[i, j] = BURNED

        self.state = new_state
        ember_step(self)

    def run(self, time_steps=50):
        for _ in range(time_steps):
            self.step()

    def compute_loss(self):
        total = np.sum(self.structures)
        burned = np.sum((self.state == BURNED) & self.structures)
        return burned / total if total > 0 else 0
