import random
import math
import cmath
import matplotlib.pyplot as plt
import time

#--------------------------------------------------- Paramètres ---------------------------------------------------#

SEED = 42 # Seed pour la génération de nombres aléatoires
PI = math.pi # Valeur de PI

PERIOD = 0.7 # Période des noyaux gaussiens
FREQUENCY = 1 / PERIOD * 2 * PI # Fréquence des noyaux gaussiens
ORIENTATION = -PI / 4 # Orientation des noyaux gaussiens
RESOLUTION = 1024 # Résolution de l'image

MAGNITUDE = 1 # Amplitude des noyaux gaussiens
BANDWITH = 2 # Taille des noyaux gaussiens
NB_KERNELS = 25 # Nombre de noyaux gaussiens
GRID_SIZE = (5, 5)  # Dimensions de la grille

random.seed(SEED)

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
            x_sample = random.uniform(x_min + i * x_step, x_min + (i + 1) * x_step)
            y_sample = random.uniform(y_min + j * y_step, y_min + (j + 1) * y_step)
            samples.append((x_sample, y_sample))
    return samples

def phasor_kernel(x, y, x0, y0, orientation, magnitude=MAGNITUDE, width=BANDWITH):
    """
    Permet de générer un noyau de phasor.
    """
    gauss = magnitude * math.exp(-PI * width**2 * ((x - x0)**2 + (y - y0)**2))
    phase = 2 * PI * FREQUENCY * ((x - x0) * math.cos(orientation) + (y - y0) * math.sin(orientation))
    return gauss * cmath.exp(1j * phase)

def phasor_noise():
    """
    Permet de générer un bruit de phasor
    """
    x = [i * 2 / (RESOLUTION - 1) - 1 for i in range(RESOLUTION)]
    y = [i * 2 / (RESOLUTION - 1) - 1 for i in range(RESOLUTION)]
    X, Y = [[0] * RESOLUTION for _ in range(RESOLUTION)], [[0] * RESOLUTION for _ in range(RESOLUTION)]
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            X[i][j] = x[j]
            Y[i][j] = y[i]
    positions = stratified_sampling(GRID_SIZE, (-1, 1), (-1, 1))
    phasor_field = [[0 + 0j] * RESOLUTION for _ in range(RESOLUTION)]
    for i in range(NB_KERNELS):
        x0, y0 = positions[i]
        for xi in range(RESOLUTION):
            for yi in range(RESOLUTION):
                phasor_field[xi][yi] += phasor_kernel(X[xi][yi], Y[xi][yi], x0, y0, ORIENTATION)
    phase_noise = [[cmath.phase(phasor_field[xi][yi]) for yi in range(RESOLUTION)] for xi in range(RESOLUTION)]
    return phase_noise

def show_phasor_noise():
    """
    Affiche le bruit de phasor généré.
    """
    start_time = time.time()
    noise = phasor_noise()
    end_time = time.time()
    execution_time = end_time - start_time

    plt.figure(figsize=(6, 6))
    plt.imshow(noise, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title(f"Phasor Noise avec {NB_KERNELS} noyaux\nExecution Time: {execution_time:.2f} seconds")
    plt.show()

if __name__ == "__main__":
    show_phasor_noise()