def apply_defensible_space(model, radius=1, reduction=0.5):
    for i in range(model.grid_size):
        for j in range(model.grid_size):
            if model.structures[i, j]:
                for di in range(-radius, radius+1):
                    for dj in range(-radius, radius+1):
                        ni, nj = i+di, j+dj
                        if 0 <= ni < model.grid_size and 0 <= nj < model.grid_size:
                            model.fuel[ni, nj] *= reduction
