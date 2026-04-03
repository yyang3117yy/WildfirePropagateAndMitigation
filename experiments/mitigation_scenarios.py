from simulation.model import WildfireModel
from simulation.mitigation import apply_defensible_space

model = WildfireModel(grid_size=100)
model.initialize()

apply_defensible_space(model)

model.run(50)

print("Loss with mitigation:", model.compute_loss())
