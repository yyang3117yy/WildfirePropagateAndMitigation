from simulation.model import WildfireModel
from visualization.plot import plot_final
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from visualization.animation import animate_fire
from visualization.animation import animate_simulation_ensemble
# ----------------------------


GRID_SIZE = 11
SIMS = 1000
TIME_STEPS = 100

UNBURNED, BURNING, BURNED = 0, 1, 2


# ----------------------------
# 1. CUSTOM INITIALIZATION
# ----------------------------
def initialize_wui_model(hardening=False, veg_mitigation=False):
    model = WildfireModel(grid_size=GRID_SIZE)

    n = GRID_SIZE

    # Initialize arrays
    model.state = np.zeros((n, n), dtype=int)
    model.fuel = np.ones((n, n)) * 0.8
    model.moisture = np.ones((n, n)) * 0.5

    model.structures = np.zeros((n, n), dtype=bool)
    model.hardening = np.zeros((n, n))

    # ----------------------------
    # LEFT: vegetation (cols 0–9)
    # ----------------------------
    # already vegetation by default

    # ----------------------------
    # RIGHT: alternating structure / vegetation (cols 10–30)
    # ----------------------------
    for j in range(1, n):
        for i in range(n):
            if (j - 1) % 2 == 0 and (i - 1) % 2 == 0:
                # structure column
                model.structures[i, j] = True
    # reset the fuel and moisture for structures
    model.fuel[model.structures] = 0.5
    model.moisture[model.structures] = 0.2
    # ----------------------------
    # Apply hardening
    # ----------------------------
    if hardening:
        model.hardening[model.structures] = 0.8  # strong hardening
    else:
        model.hardening[model.structures] = 0.2  # weak/no hardening

    # ----------------------------
    # Vegetation mitigation (fuel reduction)
    # ----------------------------
    if veg_mitigation:
        for j in range(1, n):
            model.fuel[model.structures==0]= 0.5  # reduce fuel in WUI
    
    
    

    # ----------------------------
    # Ignition point (row 16, col 1 → index [15,0])
    # ----------------------------
    model.state[5, 0] = BURNING

    return model


# ----------------------------
# 2. RUN MONTE CARLO
# ----------------------------
def run_experiment(hardening=False, veg_mitigation=False):

    burn_counts = np.zeros((GRID_SIZE, GRID_SIZE))

    for sim in range(SIMS):

        model = initialize_wui_model(
            hardening=hardening,
            veg_mitigation=veg_mitigation
        )

        model.run(TIME_STEPS)

        # Record burned cells
        burned = (model.state == BURNED)
        burn_counts += burned.astype(int)

        if sim % 100 == 0:
            print(f"Simulation {sim}")

    # Mean burn probability
    burn_prob = burn_counts / SIMS

    return burn_prob

def run_experiment_collect(hardening=False, veg_mitigation=False, collect=False):

    burn_counts = np.zeros((GRID_SIZE, GRID_SIZE))
    final_states = []

    for sim in range(SIMS):

        model = initialize_wui_model(
            hardening=hardening,
            veg_mitigation=veg_mitigation
        )

        model.run(TIME_STEPS)

        burned = (model.state == BURNED)
        burn_counts += burned.astype(int)

        # Store first 100 simulations
        if collect and sim < 100:
            final_states.append(model.state.copy())

    burn_prob = burn_counts / SIMS

    return burn_prob, final_states

# ----------------------------
# 3. PLOTTING FUNCTION
# ----------------------------
def plot_result(burn_prob, title):

    plt.figure(figsize=(6, 6))
    plt.imshow(burn_prob, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label="Burn Probability")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()




OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_and_save(burn_prob, title, filename):
    plt.figure(figsize=(6, 6))
    plt.imshow(burn_prob, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label="Burn Probability")
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")

    filepath = os.path.join(OUTPUT_DIR, f"{filename}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {filepath}")

import os

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_to_csv(matrix, filename):
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.csv")
    np.savetxt(filepath, matrix, delimiter=",")
    print(f"Saved CSV: {filepath}")

def compute_difference(map_scenario, map_baseline):
    return map_scenario - map_baseline


def plot_initial_grid(structures):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from matplotlib.colors import ListedColormap

    OUTPUT_DIR = "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Convert boolean → int (0 = vegetation, 1 = structure)
    grid = structures.astype(int)

    # ----------------------------
    # Custom colormap
    # 0 → green (vegetation)
    # 1 → grey (structure)
    # ----------------------------
    cmap = ListedColormap(["green", "grey"])

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=1)

    # Custom colorbar labels
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(["Vegetation", "Structure"])

    plt.title("Initial Grid Setup")
    plt.xlabel("Column")
    plt.ylabel("Row")

    filepath = os.path.join(OUTPUT_DIR, "initial_grid.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved initial grid plot: {filepath}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "initial_grid.csv")
    np.savetxt(csv_path, grid, delimiter=",", fmt="%d")
    print(f"Saved initial grid CSV: {csv_path}")




# ----------------------------
# 4. RUN ALL SCENARIOS
# ----------------------------
if __name__ == "__main__":

    print("Running Baseline...")
    baseline = run_experiment(False, False)
    plot_and_save(baseline, "Baseline Burn Probability", "baseline")
    save_to_csv(baseline, "baseline")

    print("Running Structure Hardening...")
    hardening = run_experiment(True, False)
    plot_and_save(hardening, "Structure Hardening", "hardening")
    save_to_csv(hardening, "hardening")

    print("Running Vegetation Mitigation...")
    veg = run_experiment(False, True)
    plot_and_save(veg, "Vegetation Mitigation", "vegetation")
    save_to_csv(veg, "vegetation")

    print("Running Combined...")
    combined = run_experiment(True, True)
    plot_and_save(combined, "Combined Mitigation", "combined")
    save_to_csv(combined, "combined")

 

    demo_model = initialize_wui_model(
        hardening=False,
        veg_mitigation=False
    )

    animate_fire(demo_model, steps=50, interval=0.2, save=True)




    # ----------------------------
    # DIFFERENCE MAPS (vs baseline)
    # ----------------------------
    print("Computing difference maps...")

    diff_hardening = compute_difference(hardening, baseline)
    diff_veg = compute_difference(veg, baseline)
    diff_combined = compute_difference(combined, baseline)

    # Plot difference maps (use diverging colormap)
    def plot_diff(map_data, title, filename):
        plt.figure(figsize=(6, 6))
        plt.imshow(map_data, cmap='bwr', vmin=-1, vmax=1)
        plt.colorbar(label="Change in Burn Probability")
        plt.title(title)
        plt.xlabel("Column")
        plt.ylabel("Row")

        filepath = os.path.join(OUTPUT_DIR, f"{filename}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved diff plot: {filepath}")

    plot_diff(diff_hardening, "Hardening - Baseline", "diff_hardening")
    plot_diff(diff_veg, "Vegetation - Baseline", "diff_vegetation")
    plot_diff(diff_combined, "Combined - Baseline", "diff_combined")

    # Save CSVs
    save_to_csv(diff_hardening, "diff_hardening")
    save_to_csv(diff_veg, "diff_vegetation")
    save_to_csv(diff_combined, "diff_combined")


    # Create one model just to extract the grid
    model = initialize_wui_model()

    # Plot and save initial grid
    plot_initial_grid(model.structures)



    baseline, final_states = run_experiment_collect(
        hardening=False,
        veg_mitigation=False,
        collect=True
    )

    # Use structures from one model (same layout)
    model = initialize_wui_model()
    
    animate_simulation_ensemble(
        final_states,
        model.structures,
        interval=0.2,
        save=True   # 👈 turn this on
    )





#model = WildfireModel(grid_size=100)
#model.initialize()
#model.run(time_steps=50)

#plot_final(model.state)
#print("Loss ratio:", model.compute_loss())
