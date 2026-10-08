"""Save checkpoint predictions in the gammaI-only output format."""
from pathlib import Path
import numpy as np


def write_prediction(model, checkpoint, output, num_cities, num_ages):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    states, ge_age, gl_age, mobility = model.predict(checkpoint)
    days, nodes, _ = states.shape
    if nodes != num_cities * num_ages:
        raise ValueError("Predicted node count does not match the scenario")
    gamma_i = states[..., -1:].astype(np.float32)
    city_gamma_i = gamma_i.reshape(days, num_cities, num_ages, 1).sum(axis=2)
    ge_city = np.zeros((days, num_cities, num_cities), dtype=np.float64)
    for source in range(num_cities):
        for target in range(num_cities):
            if source != target:
                for age in range(num_ages):
                    ge_city[:, source, target] += ge_age[:, source * num_ages + age,
                                                       target * num_ages + age]
    for day in range(days):
        np.fill_diagonal(ge_city[day], -ge_city[day].sum(axis=1))
    np.save(output / "pred_u.npy", gamma_i)
    np.save(output / "pred_u_city.npy", city_gamma_i)
    np.save(output / "GE_city.npy", ge_city)
    np.save(output / "GL_age.npy", gl_age)
    if num_cities > 1:
        np.save(output / "pred_mobility.npy", mobility)
