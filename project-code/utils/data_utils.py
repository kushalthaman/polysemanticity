import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from enum import Enum
from functools import total_ordering

def sample_geometrically(array, num_samples):
    """
    Samples the array geometrically.
    Possibly produces fewer samples than num_samples due to duplicates.
    """
    geometric_indices = np.geomspace(1, len(array), num=num_samples, dtype=int)
    geometric_indices = np.unique(geometric_indices)
    return [array[i-1] for i in geometric_indices]

def one_indexed_dataframe(matrix):
    """
    Converts a matrix to a one-indexed pandas DataFrame.
    """
    dataframe = pd.DataFrame(matrix)
    dataframe.index = range(1, len(matrix) + 1)
    dataframe.columns = range(1, len(matrix[0]) + 1)
    return dataframe

@total_ordering
class NoiseOptions(Enum):
    NONE = 1
    BERNOULLI = 2
    GAUSSIAN = 3
    def __lt__(self, other):
        return self.value < other.value

def noise_to_colour(noise):
    if noise == NoiseOptions.BERNOULLI:
        return plt.cm.Reds
    elif noise == NoiseOptions.GAUSSIAN:
        return plt.cm.Blues
    else:
        raise ValueError(f"Invalid noise option: {noise}")

def noise_to_name(noise):
    if noise == NoiseOptions.BERNOULLI:
        return "bernoulli"
    elif noise == NoiseOptions.GAUSSIAN:
        return "gaussian"
    else:
        raise ValueError(f"Invalid noise option: {noise}")

def name_to_noise(name):
    if name == "none":
        return NoiseOptions.NONE
    elif name == "bernoulli":
        return NoiseOptions.BERNOULLI
    elif name == "gaussian":
        return NoiseOptions.GAUSSIAN
    else:
        raise ValueError(f"Invalid noise name: {name}")