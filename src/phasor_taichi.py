import taichi as ti
import math, random
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)

# Paramètres du bruit
RESOLUTION = 500
FREQUENCY = 75.0
ORIENTATION = 0.5
NB_CELLS = 10

PI = 3.141592
CELL_SIZE = 2 / NB_CELLS
BANDWITH = 1.0 / CELL_SIZE

# Définition des champs Taichi 
gabor_img   = ti.field(dtype=ti.f32, shape=(RESOLUTION, RESOLUTION))
phasor_real = ti.field(dtype=ti.f32, shape=(RESOLUTION, RESOLUTION))
phasor_imag = ti.field(dtype=ti.f32, shape=(RESOLUTION, RESOLUTION))
phasor_img  = ti.field(dtype=ti.f32, shape=(RESOLUTION, RESOLUTION))

def linspace(a, b, N):
    if N <= 1:
        return [a]
    return [a + (b - a) / (N - 1) * i for i in range(N)]

# Fonction effectuant l'échantillonnage aléatoire stratifié 
def generate_grid_points():
    x_cells = linspace(-1, 1, NB_CELLS + 1)
    y_cells = linspace(-1, 1, NB_CELLS + 1)
    points = [[None for _ in range(NB_CELLS)] for _ in range(NB_CELLS)]
    for y in range(len(y_cells) - 1):
        for x in range(len(x_cells) - 1):
            kx = random.uniform(x_cells[x], x_cells[x+1])
            ky = random.uniform(y_cells[y], y_cells[y+1])
            points[NB_CELLS - y - 1][x] = (
                kx, ky,
                (x_cells[x], x_cells[x+1], RESOLUTION // NB_CELLS * x, RESOLUTION // NB_CELLS * (x+1)),
                (y_cells[y], y_cells[y+1], RESOLUTION // NB_CELLS * y, RESOLUTION // NB_CELLS * (y+1))
            )
    return points

positions = generate_grid_points()

# Génération des bruits de Phasor et de Gabor
@ti.kernel
def add_gabor_kernel(x0: ti.f32, y0: ti.f32, xmin: ti.f32, xmax: ti.f32, ymin: ti.f32, 
                     ymax: ti.f32, pymin: ti.i32, pxmin: ti.i32, height: ti.i32, width: ti.i32):
    for i, j in ti.ndrange(height, width):
        vx = xmin + j * (xmax - xmin) / (width - 1)
        vy = ymin + i * (ymax - ymin) / (height - 1)
        gaussian = ti.exp(-PI * BANDWITH * BANDWITH * ((vx - x0)**2 + (vy - y0)**2))
        sinusoid = ti.sin(FREQUENCY * ((vx - x0) * ti.cos(ORIENTATION) + (vy - y0) * ti.sin(ORIENTATION)))
        gabor_img[pymin + i, pxmin + j] += gaussian * sinusoid

@ti.kernel
def add_phasor_kernel(x0: ti.f32, y0: ti.f32, xmin: ti.f32, xmax: ti.f32, ymin: ti.f32, 
                      ymax: ti.f32, pymin: ti.i32, pxmin: ti.i32, height: ti.i32, width: ti.i32):
    for i, j in ti.ndrange(height, width):
        vx = xmin + j * (xmax - xmin) / (width - 1)
        vy = ymin + i * (ymax - ymin) / (height - 1)
        gaussian = ti.exp(-PI * BANDWITH * BANDWITH * ((vx - x0)**2 + (vy - y0)**2))
        cosin = ti.cos(FREQUENCY * ((vx - x0) * ti.cos(ORIENTATION) + (vy - y0) * ti.sin(ORIENTATION)))
        sinus = ti.sin(FREQUENCY * ((vx - x0) * ti.cos(ORIENTATION) + (vy - y0) * ti.sin(ORIENTATION)))
        phasor_real[pymin + i, pxmin + j] += gaussian * cosin
        phasor_imag[pymin + i, pxmin + j] += gaussian * sinus

@ti.kernel
def finalize_phasor():
    for i, j in ti.ndrange(RESOLUTION, RESOLUTION):
        phasor_img[i, j] = ti.sin(ti.atan2(phasor_real[i, j], phasor_imag[i, j]))

def compute_gabor_noise():
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            gabor_img[i, j] = 0.0
    for i1 in range(len(positions)):
        for i2 in range(len(positions[0])):
            c = positions[i1][i2]
            x, y = c[0], c[1]
            xmin, xmax, pxmin, pxmax = c[2]
            ymin, ymax, pymin, pymax = c[3]
            for offset in [-1, 1]:
                if (i2 + offset) >= 0 and (i2 + offset) < len(positions[0]):
                    cx = positions[i1][i2 + offset]
                    xmin = min(xmin, cx[2][0])
                    xmax = max(xmax, cx[2][1])
                    pxmin = min(pxmin, cx[2][2])
                    pxmax = max(pxmax, cx[2][3])
                if (i1 + offset) >= 0 and (i1 + offset) < len(positions):
                    cy = positions[i1 + offset][i2]
                    ymin = min(ymin, cy[3][0])
                    ymax = max(ymax, cy[3][1])
                    pymin = min(pymin, cy[3][2])
                    pymax = max(pymax, cy[3][3])
            height = pymax - pymin
            width  = pxmax - pxmin
            add_gabor_kernel(x, y, xmin, xmax, ymin, ymax, pymin, pxmin, height, width)

def compute_phasor_noise():
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            phasor_real[i, j] = 0.0
            phasor_imag[i, j] = 0.0
    for i1 in range(len(positions)):
        for i2 in range(len(positions[0])):
            c = positions[i1][i2]
            x, y = c[0], c[1]
            xmin, xmax, pxmin, pxmax = c[2]
            ymin, ymax, pymin, pymax = c[3]
            for offset in [-1, 1]:
                if (i2 + offset) >= 0 and (i2 + offset) < len(positions[0]):
                    cx = positions[i1][i2 + offset]
                    xmin = min(xmin, cx[2][0])
                    xmax = max(xmax, cx[2][1])
                    pxmin = min(pxmin, cx[2][2])
                    pxmax = max(pxmax, cx[2][3])
                if (i1 + offset) >= 0 and (i1 + offset) < len(positions):
                    cy = positions[i1 + offset][i2]
                    ymin = min(ymin, cy[3][0])
                    ymax = max(ymax, cy[3][1])
                    pymin = min(pymin, cy[3][2])
                    pymax = max(pymax, cy[3][3])
            height = pymax - pymin
            width  = pxmax - pxmin
            add_phasor_kernel(x, y, xmin, xmax, ymin, ymax, pymin, pxmin, height, width)
    finalize_phasor()

# Affichage via Matplotlib des deux bruits
def show_gabor_noise():
    compute_gabor_noise()
    _, ax = plt.subplots()
    ax.imshow(gabor_img.to_numpy().T, cmap="gray")
    plt.axis("off")
    plt.title("Bruit de gabor")
    plt.show()

def show_phasor_noise():
    compute_phasor_noise()
    _, ax = plt.subplots()
    ax.imshow(phasor_img.to_numpy().T, cmap="gray")
    plt.axis("off")
    plt.title("Bruit de phasor")
    plt.show()

if __name__ == '__main__':
    show_phasor_noise()
    show_gabor_noise()