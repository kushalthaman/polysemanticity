import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torch

from utils import *
from models import *

def train_correlations(min_m, max_m, n_instances, n_features, tied, distri, nonlin, device, l1, steps, lr):
    hidden_range = np.arange(min_m, max_m + min_m, min_m)
    weight_correlations = []
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
        starting_weights = model.W.detach().cpu()
        print(starting_weights.shape)
        train_autoencoder(model, optimizer=torch.optim.AdamW, l1=l1, steps=steps, moments=None, optimizer_kwargs={"lr": lr})
        ending_weights = model.W.detach().cpu()
        print(ending_weights.shape)
        weight_correlations.append(torch.einsum("i n m, i l m -> i n l", starting_weights, ending_weights).numpy())
        print(weight_correlations[-1].shape)

    weight_correlations = np.array(weight_correlations)
    np.save("results/hidden_range_for_correl.npy", hidden_range)
    np.save("results/weight_correlations.npy", weight_correlations)
    return hidden_range, weight_correlations

def plot_correlations(hidden_range, weight_correlations, n_features):
    for i, data in zip(hidden_range, weight_correlations):
        if i == 4096:
            mean_correl = np.mean(data, axis = 0)
            plt.figure(figsize=(10,10))
            sns.heatmap(mean_correl, cmap="viridis")
            plt.title(f"Weight Correlations for $n$ = 256, $m$ = {i}")
            plt.savefig("results/incidental_initialisations.png", bbox_inches="tight")

def main(args):
    hidden_range, weight_correlations = train_correlations(args.min_m, args.max_m, args.n_instances, args.n_features, args.tied, args.distri, args.nonlin, args.device, args.l1, args.steps, args.lr)
    hidden_range = np.load("results/hidden_range_for_correl.npy")
    weight_correlations = np.load("results/weight_correlations.npy")
    plot_correlations(hidden_range, weight_correlations, args.n_features)

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
