import math
import numpy as np

def initialise_weights(m, stdev_scaling):
    weights = np.random.normal(0, stdev_scaling/math.sqrt(m), m)
    weights = np.abs(weights)
    weights = np.sort(weights)[::-1]
    return weights

def simulate_dynamics(weights, lamb, step_size):
    metrics = {
        "t": [],
        "m_prime": [],
        "l1_norm": []
    }
    t = 0
    while len(weights) > 1:
        metrics["t"].append(t)
        metrics["m_prime"].append(len(weights))
        metrics["l1_norm"].append(np.sum(np.abs(weights)))
        t += step_size
        dw_dt = (1 - np.sum(weights ** 2)) * weights - lamb
        weights += dw_dt * step_size
        weights = weights[weights > 0.0]
    return metrics

