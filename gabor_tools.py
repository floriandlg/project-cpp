import taichi as ti
import math
import numpy as np

ti.init()

RES = 512

image = ti.field(dtype=ti.f32, shape=(RES, RES))

# Settings
frequency = ti.field(dtype=ti.f32, shape=())
orientation = ti.field(dtype=ti.f32, shape=())
nb_cells = ti.field(dtype=ti.i32, shape=())
profile = ti.field(dtype=ti.i32, shape=())

MAX_CELLS = 300
positions = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CELLS * MAX_CELLS)

def generate_points_numpy(n_cells):
    cell_size = 2.0 / n_cells
    rng = np.random.default_rng(42)
    points = []
    for i in range(n_cells):
        for j in range(n_cells):
            x = -1 + (i + 0.5) * cell_size + rng.uniform(-cell_size/2, cell_size/2)
            y = -1 + (j + 0.5) * cell_size + rng.uniform(-cell_size/2, cell_size/2)
            points.append([x, y])
    return np.array(points, dtype=np.float32)

def update_points_in_taichi(points_np):
    for i in range(points_np.shape[0]):
        positions[i] = ti.Vector([points_np[i][0], points_np[i][1]])

@ti.func
def sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

@ti.func
def apply_profile(v):
    result = v
    p = profile[None]
    if p == 0:
        result = ti.sin(v)
    elif p == 1:
        result = sign(ti.sin(v))
    elif p == 2:
        result = 2 * ti.abs(2 * (v / (2 * math.pi) - ti.floor(v / (2 * math.pi) + 0.5))) - 1
    elif p == 3:
        result = 2 * (v / (2 * math.pi) - ti.floor(v / (2 * math.pi) + 0.5))
    return result

@ti.kernel
def gabor_kernel():
    cell_size = 2.0 / nb_cells[None]
    bw = 1.0 / cell_size
    n = nb_cells[None]
    count = n * n

    for i, j in image:
        x = i / RES * 2 - 1
        y = j / RES * 2 - 1
        value = 0.0

        for k in range(count):
            pos = positions[k]
            dx = x - pos[0]
            dy = y - pos[1]
            sinusoid = ti.sin(frequency[None] * (dx * ti.cos(orientation[None]) + dy * ti.sin(orientation[None])))
            gaussian = ti.exp(-math.pi * bw**2 * (dx**2 + dy**2))
            value += gaussian * sinusoid

        image[i, j] = value

@ti.kernel
def phasor_kernel():
    cell_size = 2.0 / nb_cells[None]
    bw = 1.0 / cell_size
    n = nb_cells[None]
    count = n * n

    for i, j in image:
        x = i / RES * 2 - 1
        y = j / RES * 2 - 1
        real = 0.0
        imag = 0.0

        for k in range(count):
            pos = positions[k]
            dx = x - pos[0]
            dy = y - pos[1]

            gaussian = ti.exp(-math.pi * bw**2 * (dx**2 + dy**2))
            arg = frequency[None] * (dx * ti.cos(orientation[None]) + dy * ti.sin(orientation[None]))
            real += gaussian * ti.cos(arg)
            imag += gaussian * ti.sin(arg)

        image[i, j] = apply_profile(ti.atan2(imag, real))

# GUI Setup
gui = ti.GUI("Phasor & Gabor", res=(RES, RES))
use_gabor = False  # État initial : Phasor

# Initialisation des paramètres (ajustez selon vos besoins)
frequency[None] = 50.
orientation[None] = 90.
nb_cells[None] = 10
profile[None] = 0

# Générer les points
points_np = generate_points_numpy(nb_cells[None])
update_points_in_taichi(points_np)

while gui.running:
    # Gérer les événements
    for e in gui.get_events():
        if e.key == ti.GUI.SPACE and e.type == ti.GUI.PRESS:  # Basculer uniquement lors de l'appui
            use_gabor = not use_gabor
        elif e.key == ti.GUI.ESCAPE:  # Quitter avec ESC
            gui.running = False

    # Mettre à jour le rendu en fonction de l'état
    if use_gabor:
        gabor_kernel()
    else:
        phasor_kernel()

    # Afficher l'état actuel
    gui.text("Mode: Gabor" if use_gabor else "Mode: Phasor", pos=(0.1, 0.9))

    # Afficher l'image
    gui.set_image((image.to_numpy() + 1) / 2)
    gui.show()