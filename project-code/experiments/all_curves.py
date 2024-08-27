import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
import io
from contextlib import redirect_stdout
import re
import ast

from utils import *
from models import *

def count_polysemantic_neurons(model):
    weights = (model.W.detach().cpu().numpy())
    activations = ((np.abs(weights) >= 0.8) & (np.abs(weights) <= 1.2)).sum(axis=1)
    polysemantic = (activations > 1.1)
    return polysemantic.sum(axis=1)

def train_all_collisions(min_m, max_m, n_instances, n_features, tied, distri, nonlin, device, l1, steps, lr):
    hidden_range = np.arange(min_m, max_m + min_m, min_m)
    polysemantic_data = {'hidden_neurons': [], 'curve': [], 'polysemantic_neurons': []}
    for n_hidden in hidden_range:
        for lamb in [0.001, 0.01, 0.1]:
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
            train_autoencoder(model, optimizer=torch.optim.AdamW, l1=lamb, steps=steps, moments=None, optimizer_kwargs={"lr": lr})
            polysemantic_data['hidden_neurons'].append(n_hidden)
            polysemantic_data['curve'].append(f"l1_{lamb}")
            polysemantic_data['polysemantic_neurons'].append(count_polysemantic_neurons(model))
    print("Done with l1")
    # Add noise models
    polysemantic_data = np.array(polysemantic_data)
    np.save("results/all_curves.npy", polysemantic_data)
    return polysemantic_data

def plot_all_collisions(polysemantic_data, n_features):
    output = io.StringIO()
    with redirect_stdout(output):
        print(polysemantic_data)
    captured = output.getvalue()
    formatted = re.sub(r"array\((.*?)\)", r"\1", captured)
    dict = ast.literal_eval(formatted)

    hidden_neurons = dict['hidden_neurons']

    curves = dict['curve']

    polysemantic_neurons = dict['polysemantic_neurons']

    percentiles = [50, 33, 66, 25, 75, 20, 80, 15, 85]
    data_by_type = {}
    for x, t, y in zip(hidden_neurons, curves, polysemantic_neurons):
        if t not in data_by_type:
            data_by_type[t] = {'x': []}
        data_by_type[t]['x'].append(x)
        for percent in percentiles:
            k = f"ys{percentiles.index(percent)}"
            if k not in data_by_type[t]:
                data_by_type[t][k] = []
            data_by_type[t][k].append(np.percentile(np.array(y), percent))

    color_map = {
        'l1_0.001': 0,
        'l1_0.01': 3,
        'l1_0.1': 6,
    }

    for t, data in data_by_type.items():
        if t == 'l1_0.001' or t == 'l1_0.01' or t == 'l1_0.1':
            color = plt.cm.viridis(color_map[t]/7)
            for i, percent in enumerate(percentiles):
                plt.plot(data["x"], data[f"ys{i}"], alpha=1/(i+1), color=color, linewidth=1)
    plt.xlabel("Number of Hidden Neurons")
    plt.ylabel("Number of Polysemantic Neurons")
    plt.title(f"Polysemantic Neurons for {n_features} Features")
    plt.savefig("results/incidental_curves.png", bbox_inches="tight", dpi=300)

def main(args):
    polysemantic_data = train_all_collisions(args.min_m, args.max_m, args.n_instances, args.n_features, args.tied, args.distri, args.nonlin, args.device, args.l1, args.steps, args.lr)
    polysemantic_data = np.load("results/all_curves.npy", allow_pickle=True)
    plot_all_collisions(polysemantic_data, args.n_features)

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
