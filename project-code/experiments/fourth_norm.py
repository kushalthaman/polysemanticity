import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import pickle

from utils import *
from models import *

def list_of_strings(arg):
    return arg.split(',')

def list_of_floats(arg):
    return [float(x) for x in arg.split(',')]

def compare_noises(n_instances, n_features, n_hidden, tied, distri, nonlin, device, noise, noise_stdev, l1, steps, lr):
    norm_data = {'noise_type': [], 'noise_stdev': [], 'step': [], 'fourth_norms': []}
    set_deterministic(13)
    for type in noise:
        for stdev in noise_stdev:
            config = Config(
                n_features=n_features,
                n_hidden=n_hidden,
                n_instances=n_instances,
                tied_weights=tied,
                weight_distri=distri,
                nonlinearity=nonlin,
                noise=name_to_noise(type),
                noise_stdev=stdev
            )
            model = AutoEncoder(
                config=config,
                device=device
            )
            fourth_norms = []
            train_autoencoder(model, optimizer=torch.optim.SGD, l1=l1, steps=steps, moments=fourth_norms, optimizer_kwargs={"lr": lr})
            for step, norms in fourth_norms:
                norm_data['noise_type'].append(type)
                norm_data['noise_stdev'].append(stdev)
                norm_data['step'].append(step)
                norm_data['fourth_norms'].append(norms)
    with open(f"results/norm_by_noise.pkl", 'wb') as f:
        pickle.dump(norm_data, f)    

def plot_norm_by_noise():
    with open(f"results/norm_by_noise.pkl", 'rb') as f:
        norm_data = pickle.load(f)
    norm_data = pd.DataFrame(norm_data)
    samples = sample_geometrically(pd.unique(norm_data['step']), 50)
    norm_data_sampled = norm_data[norm_data['step'].isin(samples)]
    norm_data_sampled["average_fourth_norms"] = norm_data_sampled["fourth_norms"].apply(lambda x: np.mean(x, axis=1))
    norm_data_sampled["average_fourth_norms"] = norm_data_sampled["average_fourth_norms"].apply(lambda x: x.tolist())
    norm_data_sampled = norm_data_sampled.explode("average_fourth_norms")

    normalise = LogNorm(vmin=min(norm_data_sampled['noise_stdev'])/2, vmax=max(norm_data_sampled['noise_stdev']))

    plt.figure(figsize=(16, 10))
    for (type, stdev), group in norm_data_sampled.groupby(["noise_type", "noise_stdev"]):
        sns.lineplot(data=group, x="step", y="average_fourth_norms", label=f"{type} with $\sigma$ = {stdev}", color=noise_to_colour(name_to_noise(type))(normalise(stdev)))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Training steps")
    plt.ylabel("Average fourth norm")
    plt.title("Average fourth norm by noise")
    ax = plt.gca()
    ax.xaxis.grid()
    ax.yaxis.grid(which="both")
    plt.savefig("results/incidental_norms.png", bbox_inches="tight")

def main(args):
    compare_noises(args.n_instances, args.n_features, args.n_hidden, args.tied, args.distri, args.nonlin, args.device, args.noise, args.noise_stdev, args.l1, args.steps, args.lr)
    plot_norm_by_noise()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_instances", type=int, default=5)
    parser.add_argument("--n_features", type=int, default=8)
    parser.add_argument("--n_hidden", type=int, default=16)
    parser.add_argument("--tied", type=bool, default=True)
    parser.add_argument("--distri", type=str, default="normal")
    parser.add_argument("--nonlin", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--noise", type=list_of_strings, default=["bernoulli", "gaussian"])
    parser.add_argument("--noise_stdev", type=list_of_floats, default=[0.07, 0.1, 0.15])
    parser.add_argument("--l1", type=float, default=0)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()
    main(args)