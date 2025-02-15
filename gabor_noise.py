import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy.fft import fft2, fftshift

SEED = 42
PI = np.pi

PERIOD = 0.1
FREQUENCY = 1 / PERIOD 
ORIENTATION = -PI/4
RESOLUTION = 512

MAGNITUDE = 1
WIDTH = 2
NB_KERNELS = 25
GRID_SIZE = (5, 5)  


rng = default_rng(SEED)

def stratified_sampling(grid_size, x_range, y_range):
    num_x_cells, num_y_cells = grid_size
    x_min, x_max = x_range
    y_min, y_max = y_range

    x_step = (x_max - x_min) / num_x_cells
    y_step = (y_max - y_min) / num_y_cells

    samples = []
    for i in range(num_x_cells):
        for j in range(num_y_cells):
            x_sample = rng.uniform(x_min + i * x_step, x_min + (i + 1) * x_step)
            y_sample = rng.uniform(y_min + j * y_step, y_min + (j + 1) * y_step)
            samples.append((x_sample, y_sample))

    return samples

def gabor_kernel(x, y, x0, y0, orientation, magnitude=MAGNITUDE, width=WIDTH):
    return magnitude * np.exp(-PI * width**2 * ((x - x0)**2 + (y - y0)**2)) * np.cos(2 * PI * FREQUENCY * ((x - x0) * np.cos(orientation) + (y - y0) * np.sin(orientation)))

def gabor_noise(x, y, positions):
    x = np.linspace(-1, 1, RESOLUTION)
    y = np.linspace(-1, 1, RESOLUTION)
    X, Y = np.meshgrid(x, y)
    positions = stratified_sampling(GRID_SIZE, (-1, 1), (-1, 1))
    gabor_noise = np.zeros((RESOLUTION, RESOLUTION))
    for i in range(NB_KERNELS):
        x0, y0 = positions[i]
        gabor_noise += gabor_kernel(X, Y, x0, y0, ORIENTATION)
    return gabor_noise

def show_gabor_noise():
    gabor_noise_show = gabor_noise(RESOLUTION, RESOLUTION, GRID_SIZE)
    plt.figure(figsize=(6, 6))
    plt.imshow(gabor_noise_show,extent=(-1,1,-1,1),cmap='gray',origin='lower')
    plt.title(f"Gabor Noise avec {NB_KERNELS} noyaux")
    plt.show()

def show_gabor_noise_fft():
    x = np.linspace(-1, 1, RESOLUTION)
    y = np.linspace(-1, 1, RESOLUTION)
    gabor_noise_lst = gabor_noise(RESOLUTION, RESOLUTION, GRID_SIZE)
    gabor_noise_fft = fftshift(fft2(gabor_noise_lst))
    plt.imshow(np.abs(gabor_noise_fft), extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title("Fourier Transform of Gabor Kernel")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.show()

if __name__ == "__main__":
    show_gabor_noise()