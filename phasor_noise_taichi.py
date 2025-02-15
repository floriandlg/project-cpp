import taichi as ti
import matplotlib.pyplot as plt
import time

# Initialize Taichi
ti.init(arch=ti.cpu)

#--------------------------------------------------- Paramètres ---------------------------------------------------#

SEED = 42 # Seed pour la génération de nombres aléatoires
PI = 3.141592653589793 # Valeur de PI

PERIOD = 0.7 # Période des noyaux gaussiens
FREQUENCY = 1 / PERIOD * 2 * PI # Fréquence des noyaux gaussiens
ORIENTATION = -PI / 4 # Orientation des noyaux gaussiens
RESOLUTION = 1024 # Résolution de l'image

MAGNITUDE = 1 # Amplitude des noyaux gaussiens
BANDWITH = 2 # Taille des noyaux gaussiens
NB_KERNELS = 25 # Nombre de noyaux gaussiens
GRID_SIZE = (5, 5)  # Dimensions de la grille

#--------------------------------------------------- Paramètres ---------------------------------------------------#

@ti.kernel
def stratified_sampling(samples: ti.template(), grid_size_x: int, grid_size_y: int, x_min: float, x_max: float, y_min: float, y_max: float): # type: ignore
    x_step = (x_max - x_min) / grid_size_x
    y_step = (y_max - y_min) / grid_size_y
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            x_sample = ti.random() * x_step + x_min + i * x_step
            y_sample = ti.random() * y_step + y_min + j * y_step
            samples[i * grid_size_y + j] = [x_sample, y_sample]

@ti.func
def phasor_kernel(x, y, x0, y0, orientation, magnitude=MAGNITUDE, width=BANDWITH):
    gauss = magnitude * ti.exp(-PI * width**2 * ((x - x0)**2 + (y - y0)**2))
    phase = 2 * PI * FREQUENCY * ((x - x0) * ti.cos(orientation) + (y - y0) * ti.sin(orientation))
    return gauss * ti.Vector([ti.cos(phase), ti.sin(phase)])

@ti.kernel
def phasor_noise(phasor_field: ti.template(), phase_noise: ti.template(), positions: ti.template()): # type: ignore
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            x = -1 + 2 * i / (RESOLUTION - 1)
            y = -1 + 2 * j / (RESOLUTION - 1)
            phasor_field[i, j] = [0.0, 0.0]
            for k in range(NB_KERNELS):
                x0, y0 = positions[k]
                phasor_field[i, j] += phasor_kernel(x, y, x0, y0, ORIENTATION)
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            phase_noise[i, j] = ti.atan2(phasor_field[i, j][1], phasor_field[i, j][0])

def show_phasor_noise():
    start_time = time.time()
    
    positions = ti.Vector.field(2, dtype=ti.f32, shape=(GRID_SIZE[0] * GRID_SIZE[1]))
    stratified_sampling(positions, GRID_SIZE[0], GRID_SIZE[1], -1, 1, -1, 1)
    
    phasor_field = ti.Vector.field(2, dtype=ti.f32, shape=(RESOLUTION, RESOLUTION))
    phase_noise = ti.field(dtype=ti.f32, shape=(RESOLUTION, RESOLUTION))
    
    phasor_noise(phasor_field, phase_noise, positions)
    
    noise_np = phase_noise.to_numpy()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    plt.figure(figsize=(6, 6))
    plt.imshow(noise_np, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title(f"Phasor Noise avec {NB_KERNELS} noyaux\nExecution Time: {execution_time:.2f} seconds")
    plt.show()

if __name__ == "__main__":
    show_phasor_noise()