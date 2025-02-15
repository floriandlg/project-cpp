import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy.fft import fft2, fftshift
import time

#--------------------------------------------------- Paramètres ---------------------------------------------------#

SEED = 42 # Seed pour la génération de nombres aléatoires
PI = np.pi # Valeur de PI
PERIOD = 0.7 # Période des noyaux gaussiens
FREQUENCY = 1 / PERIOD * 2 * PI # Fréquence des noyaux gaussiens
RESOLUTION = 1024 # Résolution de l'image
MAGNITUDE = 1 # Amplitude des noyaux gaussiens (à demander explication)
BANDWITH = 2 # Taille des noyaux gaussiens
GRID_SIZE = (5, 5)  # Dimensions de la grille

ORIENTATION = -PI / 4 # Orientation des noyaux gaussiens
NB_KERNELS = 25 # Nombre de noyaux gaussiens

rng = default_rng(SEED)

#--------------------------------------------------- Paramètres ---------------------------------------------------#

def stratified_sampling(grid_size, x_range, y_range):
    """
    Génère des points aléatoires (de manière stratifiée) dans une grille 2D.
    Retourne une liste de points (x, y) générés aléatoirement dans la grille donnée par l'utilisateur.
    """
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

def phasor_kernel(x, y, x0, y0, orientation, magnitude=MAGNITUDE, width=BANDWITH):
    """
    Permet de générer un noyau de phasor.
    """
    gauss = magnitude * np.exp(-PI * width**2 * ((x - x0)**2 + (y - y0)**2))
    phase = 2 * PI * FREQUENCY * ((x - x0) * np.cos(orientation) + (y - y0) * np.sin(orientation))
    return gauss * np.exp(1j * phase)

def phasor_noise(x, y, positions):
    """
    Permet de générer un bruit de phasor
    """
    x = np.linspace(-1, 1, RESOLUTION)
    y = np.linspace(-1, 1, RESOLUTION)
    X, Y = np.meshgrid(x, y)
    positions = stratified_sampling(GRID_SIZE, (-1, 1), (-1, 1))
    phasor_field = np.zeros((RESOLUTION, RESOLUTION), dtype=np.complex128)
    for i in range(NB_KERNELS):
        x0, y0 = positions[i]
        phasor_field += phasor_kernel(X, Y, x0, y0, ORIENTATION)
    phase_noise = np.angle(phasor_field)
    return phase_noise

def show_phasor_noise():
    """
    Affiche le bruit de phasor généré.
    """
    start_time = time.time()
    noise = phasor_noise(RESOLUTION, RESOLUTION, GRID_SIZE)
    end_time = time.time()
    execution_time = end_time - start_time

    plt.figure(figsize=(6, 6))
    plt.imshow(noise, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title(f"Phasor Noise avec {NB_KERNELS} noyaux\nExecution Time: {execution_time:.2f} seconds")
    plt.show()

def show_phasor_noise_fft():
    """
    Affiche la transformée de Fourier du bruit de phasor généré.
    """
    start_time = time.time()
    x = np.linspace(-1, 1, RESOLUTION)
    y = np.linspace(-1, 1, RESOLUTION)
    noise = phasor_noise(RESOLUTION, RESOLUTION, GRID_SIZE)
    noise_fft = fftshift(fft2(noise))
    end_time = time.time()
    execution_time = end_time - start_time

    plt.imshow(np.abs(noise_fft), extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title(f"Fourier Transform of Phasor Noise\nExecution Time: {execution_time:.2f} seconds")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.show()

if __name__ == "__main__":
    show_phasor_noise()
    show_phasor_noise_fft()