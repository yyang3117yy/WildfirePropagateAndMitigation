import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import imageio
import os

def animate(model, steps=50):
    plt.ion()
    for _ in range(steps):
        plt.clf()
        plt.imshow(model.state, cmap='hot')
        plt.title("Wildfire Spread")
        plt.pause(0.1)
        model.step()
    plt.ioff()
    plt.show()



# State definitions
UNBURNED, BURNING, BURNED = 0, 1, 2

def animate_fire(model, steps=50, interval=0.2, save=False):
    """
    Animate wildfire spread over time.

    Parameters:
    - model: initialized WildfireModel
    - steps: number of time steps
    - interval: pause between frames (seconds)
    - save: whether to save animation as GIF
    """

    # ----------------------------
    # Build visualization grid
    # ----------------------------
    def build_visual_grid():
        """
        Combine structure + fire state into one grid for plotting
        """
        grid = np.zeros_like(model.state)

        for i in range(model.grid_size):
            for j in range(model.grid_size):

                if model.state[i, j] == BURNING:
                    grid[i, j] = 2  # burning (red)

                elif model.state[i, j] == BURNED:
                    grid[i, j] = 3  # burned (black)

                else:
                    if model.structures[i, j]:
                        grid[i, j] = 1  # structure (grey)
                    else:
                        grid[i, j] = 0  # vegetation (green)

        return grid

    # ----------------------------
    # Custom colormap
    # ----------------------------
    cmap = ListedColormap([
        "green",   # 0 vegetation
        "grey",    # 1 structure
        "red",     # 2 burning
        "black"    # 3 burned
    ])

    plt.figure(figsize=(6, 6))
    plt.ion()

    frames = []

    for t in range(steps):

        plt.clf()

        vis_grid = build_visual_grid()

        plt.imshow(vis_grid, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
        plt.title(f"Wildfire Spread (Step {t})")
        plt.xlabel("Column")
        plt.ylabel("Row")

        plt.pause(interval)

        if save:
            frames.append(vis_grid.copy())

        model.step()

    plt.ioff()
    plt.show()

    # ----------------------------
    # Optional: Save GIF
    # ----------------------------
    if save:
        images = []
        for frame in frames:
            images.append((frame * 85).astype(np.uint8))  # scale for visibility

        imageio.mimsave("results/fire_animation.gif", images, duration=interval)

        print("Saved animation: results/fire_animation.gif")
    
def animate_simulation_ensemble(states_list, structures, interval=0.3, save=False):
    """
    Animate final states across multiple simulations.

    Parameters:
    - states_list: list of final state grids
    - structures: structure mask
    - interval: time between frames (seconds)
    - save: whether to save animation as GIF
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.colors import ListedColormap

    UNBURNED, BURNING, BURNED = 0, 1, 2

    OUTPUT_DIR = "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------
    # Color map
    # ----------------------------
    cmap = ListedColormap([
        "green",   # vegetation
        "grey",    # structure
        "red",     # burning
        "black"    # burned
    ])

    # ----------------------------
    # Build visualization grid
    # ----------------------------
    def build_visual(state):
        grid = np.zeros_like(state)

        for i in range(state.shape[0]):
            for j in range(state.shape[1]):

                if state[i, j] == BURNED:
                    grid[i, j] = 3
                elif state[i, j] == BURNING:
                    grid[i, j] = 2
                else:
                    if structures[i, j]:
                        grid[i, j] = 1
                    else:
                        grid[i, j] = 0

        return grid

    plt.figure(figsize=(6, 6))
    plt.ion()

    frames = []

    for idx, state in enumerate(states_list):

        plt.clf()

        vis = build_visual(state)

        plt.imshow(vis, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
        plt.title(f"Simulation {idx+1} Final State")
        plt.xlabel("Column")
        plt.ylabel("Row")

        plt.pause(interval)

        if save:
            frames.append(vis.copy())

    plt.ioff()
    plt.show()

    # ----------------------------
    # Save GIF
    # ----------------------------
    if save:
        import imageio

        images = []
        for frame in frames:
            # Scale values to 0–255 for image
            img = (frame * 85).astype(np.uint8)
            images.append(img)

        gif_path = os.path.join(OUTPUT_DIR, "ensemble_animation.gif")
        imageio.mimsave(gif_path, images, duration=interval)

        print(f"Saved animation: {gif_path}")