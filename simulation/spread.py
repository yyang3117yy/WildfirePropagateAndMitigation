import numpy as np

def compute_spread_prob(i, j, ni, nj, fuel, moisture, structures, hardening):
    base = 0.3

    fuel_factor = fuel[ni, nj]
    moisture_factor = (1 - moisture[ni, nj])

    direction = np.array([ni - i, nj - j])
    wind_direction = np.array([0, 1])
    wind_strength = 1.5

    wind_alignment = np.dot(direction, wind_direction)
    wind_factor = 1 + wind_strength * max(0, wind_alignment)

    if structures[ni, nj]:
        mitigation = (1 - hardening[ni, nj])
    else:
        mitigation = 1.0

    prob = base * fuel_factor * moisture_factor * wind_factor * mitigation
    return min(prob, 1.0)
