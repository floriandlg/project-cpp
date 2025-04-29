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
gui = ti.GUI("Phasor Noise", res=(RES, RES))
slider_freq = gui.slider("Frequence", 1, 100)
slider_orient = gui.slider("Orientation", 0, 180)
slider_cells = gui.slider("Nb cellules", 1, 15)
slider_profile = gui.slider("Profil (0:sin,1:car,2:tri,3:saw)", 0, 3)

slider_freq.value = 50
slider_orient.value = 90
slider_cells.value = 5
slider_profile.value = 0

# Valeurs précédentes
prev_freq = -1.0
prev_orient = -1.0
prev_cells = -1
prev_profile = -1

while gui.running:
    # Lire sliders
    freq_val = slider_freq.value
    orient_val = slider_orient.value * math.pi / 180
    cells_val = int(slider_cells.value)
    profile_val = int(slider_profile.value)

    update = False

    if cells_val != prev_cells:
        nb_cells[None] = cells_val
        np_points = generate_points_numpy(cells_val)
        update_points_in_taichi(np_points)
        prev_cells = cells_val
        update = True

    if (freq_val != prev_freq or
        orient_val != prev_orient or
        profile_val != prev_profile):
        frequency[None] = freq_val
        orientation[None] = orient_val
        profile[None] = profile_val
        prev_freq = freq_val
        prev_orient = orient_val
        prev_profile = profile_val
        update = True

    if update:
        phasor_kernel()

    gui.set_image((image.to_numpy() + 1) / 2)
    gui.show()
