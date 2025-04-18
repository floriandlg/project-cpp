import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

SEED = 8
GRID_SIZE = 15

rng = default_rng(SEED)

def random_sampling(grid_size:tuple, x_range:tuple, y_range:tuple):
    num_x_cells, num_y_cells = grid_size
    x_min, x_max = x_range
    y_min, y_max = y_range

    x_step = (x_max - x_min) / num_x_cells
    y_step = (y_max - y_min) / num_y_cells

    samples = []
    
    for i in range(num_x_cells):
        for j in range(num_y_cells):
            x_sample = rng.uniform(x_min, x_max)
            y_sample = rng.uniform(y_min, y_max)

            samples.append((x_sample, y_sample))

    return samples, x_step, y_step

def show_random_sampling(grid_size, x_range, y_range, ax=None):
    samples, x_step, y_step = random_sampling(grid_size, x_range, y_range)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(grid_size[0] + 1):
        ax.axvline(x=x_range[0] + i * x_step, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')

    for j in range(grid_size[1] + 1):
        ax.axhline(y=y_range[0] + j * y_step, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')

    x_vals, y_vals = zip(*samples)
    ax.scatter(x_vals, y_vals, color='blue', label='Random Samples')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Échantillonnage aléatoire")
    ax.legend()
    ax.set_aspect('equal')

def stratified_sampling(grid_size:tuple, x_range:tuple, y_range:tuple):
    num_x_cells, num_y_cells = grid_size
    x_min, x_max = x_range
    y_min, y_max = y_range

    x_step = (x_max - x_min) / num_x_cells
    y_step = (y_max - y_min) / num_y_cells

    samples = []
    
    for i in range(num_x_cells):
        for j in range(num_y_cells):
            x_cell_min = x_min + i * x_step
            x_cell_max = x_min + (i + 1) * x_step
            y_cell_min = y_min + j * y_step
            y_cell_max = y_min + (j + 1) * y_step

            x_sample = rng.uniform(x_cell_min, x_cell_max)
            y_sample = rng.uniform(y_cell_min, y_cell_max)

            samples.append((x_sample, y_sample))

    return samples,x_step,y_step

def show_stratified_sampling(grid_size, x_range, y_range, ax=None):
    samples, x_step, y_step = stratified_sampling(grid_size, x_range, y_range)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(grid_size[0] + 1):
        ax.axvline(x=x_range[0] + i * x_step, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')

    for j in range(grid_size[1] + 1):
        ax.axhline(y=y_range[0] + j * y_step, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')

    x_vals, y_vals = zip(*samples)
    ax.scatter(x_vals, y_vals, color='red', label='Stratified Samples')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Échantillonnage stratifié")
    ax.legend()
    ax.set_aspect('equal')

def show_combined_sampling(grid_size, x_range, y_range):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    show_random_sampling(grid_size, x_range, y_range, ax=ax1)
    ax1.set_title("Échantillonnage aléatoire")

    show_stratified_sampling(grid_size, x_range, y_range, ax=ax2)
    ax2.set_title("Échantillonnage stratifié")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_combined_sampling((GRID_SIZE,GRID_SIZE), (-1, 1), (-1, 1))
