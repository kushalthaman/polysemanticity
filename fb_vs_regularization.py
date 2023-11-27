from math import *

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# Compute bounds on the relative variance for each value of m' between 1 and m
def rel_var_range(my_w):
    sum_w = np.cumsum(my_w)
    sum_w_squared = np.cumsum(my_w ** 2)
    m_prime_range = np.arange(1, len(my_w) + 1)
    avg_w = sum_w / m_prime_range
    variance_w = sum_w_squared / m_prime_range - avg_w ** 2
    high_relative_variance = variance_w / (avg_w - my_w) ** 2
    low_relative_variance = variance_w / (avg_w - np.append(my_w[1:], 0)) ** 2
    return low_relative_variance, high_relative_variance


np.random.seed(0)

m = 10 ** 5
lamb = 1e-5

# Initialize w
# w = np.random.normal(0, 1, m)
# w /= sqrt(w.dot(w))
w = np.random.normal(0, .9/sqrt(m), m)
w = abs(w)
w = np.sort(w)[::-1]
print(w)
print(f"overall average: {np.average(w)}")
print(f"overall variance: {np.var(w)}")
print(f"overall relative variance: {np.var(w) / np.average(w)**2}")

w_init = w

# Computations (both theoretical and actual) to check if the relative variance will always be a constant
low_relative_variance_w, high_relative_variance_w = rel_var_range(w)
percentiles = 0.5 + (np.arange(m) + 0.5) / (2 * m)
w_ideal = norm.ppf(percentiles)[::-1]
low_relative_variance_ideal, high_relative_variance_ideal = rel_var_range(w_ideal)

printing_frequency = 3
print_at_this_step = True
m_prime_history = []
t_history = []
one_norm_history = []
feature_benefit_history = []
total_squared_deviation_history = []
relative_variance_history = []
w_history = []
gamma_history = []

w0 = 1 / sum(w)
print(f"w0: {w0}")
print(f"w0 * sqrt(m): {w0 * sqrt(m)}")

gamma = (w[0] - w[1]) / w[0]


# Training loop
t = 0.0
cnt = 0
while len(w) > 1:
    sq_norm = sum(w ** 2)
    dw_dt = (1 - sq_norm) * w - lamb
    feature_benefit = 1 - sq_norm
    one_norm = sum(w)
    delta = feature_benefit - lamb * one_norm
    average = one_norm / len(w)
    total_squared_deviation = sq_norm - one_norm ** 2 / len(w)
    variance = total_squared_deviation / len(w)
    relative_variance = variance / average ** 2

    print_at_this_step = int(log(cnt + 1) * printing_frequency) != int(log(cnt + 2) * printing_frequency)

    step_size = .5

    if print_at_this_step:
        print(f"cnt: {cnt}")
        print(f"m': {len(w)}")
        print(f"t: {t}")
        print(f"step size: {step_size}")
        print("")
        print(f"squared norm: {w.dot(w)}")
        print(f"feature benefit: {feature_benefit}")
        print(f"lambda*one-norm: {lamb * one_norm}")
        print(f"delta: {delta}")
        print(f"old value: {w}")
        print(f"gradient: {dw_dt}")
        print()

    gamma += lamb / w[0] * step_size * gamma
    w += dw_dt * step_size
    w = w[w > 0.0]  # pop zero values off

    if print_at_this_step or len(w) == 1:
        t_history.append(t)
        m_prime_history.append(len(w))
        one_norm_history.append(one_norm)
        feature_benefit_history.append(feature_benefit)
        total_squared_deviation_history.append(total_squared_deviation)
        w_history.append(w)
        gamma_history.append(gamma)
        relative_variance_history.append(relative_variance)

    cnt += 1
    t += step_size

m_prime_history = np.array(m_prime_history)


# Plots
plt.cla()
plt.clf()

plt.figure(figsize=(6, 5))

plt.loglog(t_history, m_prime_history, color="blue", label="m'")
plt.loglog(t_history, np.maximum(1 / (1 / sqrt(m) + lamb * np.array(t_history)) ** 2, 1), '--', color="cyan",
           label="predicted m'")
plt.loglog(t_history, one_norm_history, color="red", label="||W_i||_1")
plt.loglog(t_history, np.maximum(1 / (1 / sqrt(m) + lamb * np.array(t_history)), 1), '--', color="pink",
           label="predicted ||W_i||_1")

other_data = False

if other_data:
    plt.loglog(t_history, feature_benefit_history, 'o-', color="green", label="feature benefit")
    plt.loglog(t_history, lamb * np.array(one_norm_history), color="purple", label="lambda * 1-norm")
    plt.loglog(t_history, relative_variance_history, color="orange", label="relative variance")
    plt.loglog(t_history, [w[0] / (1 / sqrt(m) + t * lamb) for (w, t) in zip(w_history, t_history)], color="pink",
               label="w1 / (\"t * lambda\")")
    plt.loglog(t_history, [(sqrt(log(m)) - sqrt(log(m/m_prime))) / (sqrt(log(m / (m_prime / 2))) - sqrt(log(m / m_prime)))
                           for m_prime in m_prime_history], color="gray",
               label="theoretical w1 / (\"t * lambda)\"")
    plt.loglog(t_history, [(w_init[0] - w_init[m_prime-1]) / (w_init[(m_prime - 1) // 2] - w_init[m_prime-1])
                           for m_prime in m_prime_history], color="black",
               label="initial (w1 - smallest) / (median - smallest)")
    plt.loglog(t_history, 1 / sqrt(m) + np.array(t_history) * lamb, '--', color="pink",
               label="\"t * lamb\"")
    plt.loglog(t_history, [w[(m_prime - 1) // 2] - w[m_prime-1]
                           for w, m_prime in zip(w_history, m_prime_history)], '--', color="black",
               label="median - smallest")

# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.legend()
# plt.tight_layout()
plt.xlabel('Training time')
plt.title('Dynamics of sparsification (m = 10^5, λ = 10^-5)')
plt.grid(True)
plt.xlim(1, sqrt(10) / lamb)
# plt.ylim(lamb / 10, m * 10)
plt.ylim(1 / sqrt(10), m * sqrt(10))
plt.subplots_adjust(bottom=0.15)
plt.show()

# plt.clf()
# plt.cla()

# fig, ax1 = plt.subplots()
#
# color = 'tab:orange'
# ax1.semilogx(t_history, relative_variance_history, 'o-', color=color)
# ax1.semilogx(t_history, low_relative_variance_w[m_prime_history - 1], '-', color="red")
# ax1.semilogx(t_history, high_relative_variance_w[m_prime_history - 1], '-', color="red")
# ax1.semilogx(t_history, low_relative_variance_ideal[m_prime_history - 1], '--', color="pink")
# ax1.semilogx(t_history, high_relative_variance_ideal[m_prime_history - 1], '--', color="pink")
# ax1.grid(True, axis='x')
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.grid(True, which="major", axis='y', linestyle="--", linewidth=0.5, color=color)
# ax2 = ax1.twinx()
#
# color = 'tab:blue'
# ax2.set_ylabel("m'", color=color)
# ax2.semilogx(t_history, m_prime_history, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.grid(True, which="major", axis='y', linestyle="--", linewidth=0.5, color=color)
# plt.show()

other_plots = False

if other_plots:
    plt.clf()
    plt.cla()
    plt.xlabel('Time')
    plt.title('Relative variance')
    plt.semilogx(t_history, relative_variance_history, 'o-', color="orange")
    plt.semilogx(t_history, low_relative_variance_w[m_prime_history - 1], '-', color="red")
    plt.semilogx(t_history, high_relative_variance_w[m_prime_history - 1], '-', color="red")
    plt.semilogx(t_history, low_relative_variance_ideal[m_prime_history - 1], '--', color="pink")
    plt.semilogx(t_history, high_relative_variance_ideal[m_prime_history - 1], '--', color="pink")
    plt.grid(True)
    plt.show()

    plt.clf()
    plt.cla()
    plt.xlabel('Time')
    plt.title('Relative weights as a fraction of the biggest weight')
    plt.xscale("log")
    for t, w in zip(t_history, w_history):
        plt.scatter([t] * len(w), w / max(w), marker='x')
    plt.plot(t_history, 1 - np.array(gamma_history))
    plt.grid(True)
    plt.show()
