import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

from utils import *
from models import *

def count_polysemantic_neurons(model):
    weights = (model.W.detach().cpu().numpy())
    activations = ((np.abs(weights) >= 0.8) & (np.abs(weights) <= 1.2)).sum(axis=1)
    polysemantic = (activations > 1.1)
    return polysemantic.sum(axis=1)

def train_collisions(min_m, max_m, n_instances, n_features, tied, distri, nonlin, device, l1, steps, lr):
    hidden_range = np.arange(min_m, max_m + min_m, min_m)
    polysemantic_neurons = []
    for n_hidden in hidden_range:
        config = Config(
            n_instances = n_instances,
            n_features = n_features,
            n_hidden = n_hidden,
            tied_weights=tied,
            weight_distri=distri,
            nonlinearity=nonlin,
            noise=NoiseOptions.NONE,
            noise_stdev=0.0
        )
        model = AutoEncoder(config, device=device)
        train_autoencoder(model, optimizer=torch.optim.AdamW, l1=l1, steps=steps, moments=None, optimizer_kwargs={"lr": lr})
        polysemantic_neurons.append(count_polysemantic_neurons(model))
    polysemantic_neurons = np.array(polysemantic_neurons)
    np.save("results/hidden_range.npy", hidden_range)
    np.save("results/polysemantic_neurons.npy", polysemantic_neurons)
    return hidden_range, polysemantic_neurons

def plot_collisions(hidden_range, polysemantic_neurons, n_features):
    log_means = np.log(np.mean(polysemantic_neurons, axis=1) + 1e-5)
    log_stdevs = np.log(np.std(polysemantic_neurons, axis=1) + 1e-5)

    x_line = np.linspace(hidden_range[0], hidden_range[-1], 500)
    y_line = n_features**2 / (4 * x_line)
    log_x_line = np.log(x_line)
    log_y_line = np.log(y_line)

    plt.figure(figsize=(16, 10))
    plt.errorbar(np.log(hidden_range), log_means, yerr=log_stdevs, fmt='o', ecolor='red', capsize=5, capthick=2)

    plt.plot(log_x_line, log_y_line, label='$n^2/4m$', color='black', linestyle='--')

    plt.xlabel("Log(Number of Hidden Neurons)")
    plt.ylabel("Log(Number of Polysemantic Neurons)")
    plt.title(f"Polysemantic Neurons for n = {n_features}")
    plt.legend()
    plt.grid()
    plt.savefig("results/incidental_collisions.png", bbox_inches="tight")

def main(args):
    hidden_range, polysemantic_neurons = train_collisions(args.min_m, args.max_m, args.n_instances, args.n_features, args.tied, args.distri, args.nonlin, args.device, args.l1, args.steps, args.lr)
    plot_collisions(hidden_range, polysemantic_neurons, args.n_features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_m", type=int, default=256)
    parser.add_argument("--max_m", type=int, default=4096)
    parser.add_argument("--n_instances", type=int, default=16)
    parser.add_argument("--n_features", type=int, default=256)
    parser.add_argument("--tied", type=bool, default=True)
    parser.add_argument("--distri", type=str, default="normal")
    parser.add_argument("--nonlin", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--l1", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=7500)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    main(args)