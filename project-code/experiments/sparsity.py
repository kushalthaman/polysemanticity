
import matplotlib.pyplot as plt
import numpy as np
import argparse

from models import *

import math
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_sparsity(metrics, m, lamb):
    plt.figure(figsize=(16, 10))
    plt.loglog(metrics["t"], metrics["m_prime"], color="blue", label="$m'$")
    plt.loglog(metrics["t"], np.maximum(1 / (1 / np.sqrt(m) + lamb * np.array(metrics["t"])) ** 2, 1), '--', color="cyan", label="predicted $m'$")
    plt.loglog(metrics["t"], metrics["l1_norm"], color="red", label="$\ell_1$ norm")
    plt.loglog(metrics["t"], np.maximum(1 / (1 / np.sqrt(m) + lamb * np.array(metrics["t"])), 1), '--', color="pink", label="predicted $\ell_1$ norm")
    plt.xlabel("Training steps")
    plt.ylabel("Magnitude")
    plt.title(f"Sparsity Dynamics for $m$ = {m:.1E} and $\lambda$ = {lamb:.1E}")
    plt.legend()
    plt.grid()
    plt.savefig("results/incidental_sparsity.png", bbox_inches="tight")

def main(m, lamb):
    weights = initialise_weights(int(m), 0.9)
    metrics = simulate_dynamics(weights, lamb, 0.5)
    plot_sparsity(metrics, m, lamb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=float, default=1e5)
    parser.add_argument("--lamb", type=float, default=1e-5)
    args = parser.parse_args()
    main(args.m, args.lamb)

