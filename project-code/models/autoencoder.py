from dataclasses import dataclass
import time
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import NoiseOptions

@dataclass
class Config():
    n_instances: int
    n_features: int
    n_hidden: int
    tied_weights: bool
    weight_distri: str
    nonlinearity: bool
    noise: NoiseOptions
    noise_stdev: float

class AutoEncoder(nn.Module):
    def __init__(self, config: Config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.existing_batch = None
        if config.tied_weights:
            if config.weight_distri == "normal":
                self.W = nn.Parameter(nn.init.normal_(torch.empty((config.n_instances, config.n_features, config.n_hidden), device=device), mean=0, std=1/np.sqrt(config.n_hidden)))
            else:
                self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty((config.n_instances, config.n_features, config.n_hidden), device=device)))
        else:
            if config.weight_distri == "normal":
                self.W_enc = nn.Parameter(nn.init.normal_(torch.empty((config.n_instances, config.n_features, config.n_hidden), device=device), mean=0, std=1/np.sqrt(config.n_hidden)))
                self.W_dec = nn.Parameter(nn.init.normal_(torch.empty((config.n_instances, config.n_features, config.n_hidden), device=device), mean=0, std=1/np.sqrt(config.n_hidden)))
            else:
                self.W_enc = nn.Parameter(nn.init.xavier_normal_(torch.empty((config.n_instances, config.n_features, config.n_hidden), device=device)))
                self.W_dec = nn.Parameter(nn.init.xavier_normal_(torch.empty((config.n_instances, config.n_features, config.n_hidden), device=device)))
        self.existing_batch = None
    
    def forward(self, features):
        if self.config.tied_weights:
            hidden = torch.einsum("...ibf, ifh -> ...ibh", features, self.W)
        else:
            hidden = torch.einsum("...ibf, ifh -> ...ibh", features, self.W_enc)
        if self.config.noise == NoiseOptions.BERNOULLI:
            noise = torch.randint(0, 2, (self.config.n_hidden, ), device=self.device)
            noise = noise * 2 * self.config.noise_stdev - self.config.noise_stdev
        elif self.config.noise == NoiseOptions.GAUSSIAN:
            noise = torch.randn((self.config.n_hidden, ), dtype=torch.float, device=self.device)
            noise = noise * self.config.noise_stdev
        else:
            noise = torch.zeros((self.config.n_hidden, ), dtype=torch.float, device=self.device)
        hidden = hidden + noise
        if self.config.tied_weights:
            output = torch.einsum("...ibh, ifh -> ...ibf", hidden, self.W)
        else:
            output = torch.einsum("...ibh, ifh -> ...ibf", hidden, self.W_dec)
        if self.config.nonlinearity:
            output = F.relu(output)
        return output
    
    def generate_batch(self):
        if self.existing_batch is None:
            identity = torch.eye(self.config.n_features, device=self.device)
            self.existing_batch = identity.expand(self.config.n_instances, -1, -1)
        return self.existing_batch
    
def train_autoencoder(model, optimizer, steps, l1, moments = None, optimizer_kwargs={}):
    cfg = model.config
    opt = optimizer(list(model.parameters()), **optimizer_kwargs)
    with trange(steps) as t_steps:
        for _ in t_steps:
            opt.zero_grad()
            batch = model.generate_batch()
            output = model(batch)
            loss = F.mse_loss(output, batch, reduction='sum')
            if l1 > 0:
                if cfg.tied_weights:
                    loss += l1 * torch.norm(model.W, p=1)
                else:
                    loss += l1 * torch.norm(model.W_enc, p=1) + l1 * torch.norm(model.W_dec, p=1)
            if _ % 1000 == 0:
                print(f"Loss: {loss.item()}")
            loss.backward()
            opt.step()
            if moments is not None and ((_ % 5 == 0) or (_ == steps - 1)):
                weights = model.W.detach().cpu().numpy()
                fourth = ((weights)**4).sum(axis=2)
                moments.append((_, fourth))
