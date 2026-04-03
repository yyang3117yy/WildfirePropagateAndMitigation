import numpy as np

def ember_step(model):
    wind_direction = np.array([0, 1])
    burning_cells = np.argwhere(model.state == 1)

    for (i, j) in burning_cells:
        if np.random.rand() < 0.3:
            distance = int(np.random.exponential(scale=5))
            ni = int(i + wind_direction[0] * distance)
            nj = int(j + wind_direction[1] * distance)

            if 0 <= ni < model.grid_size and 0 <= nj < model.grid_size:
                if model.state[ni, nj] == 0:
                    ignite_prob = 0.4 * (1 - model.moisture[ni, nj])

                    if model.structures[ni, nj]:
                        ignite_prob *= (1 - model.hardening[ni, nj])

                    if np.random.rand() < ignite_prob:
                        model.state[ni, nj] = 1
