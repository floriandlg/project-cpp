import taichi as ti
import math, random

ti.init(arch=ti.cpu)


def linspace(a,b,N):
    return [a + (b-a)/(N-1)*i for i in range(N)]


class Noise:

    def __init__(self, RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS):
        self.RESOLUTION = RESOLUTION
        self.FREQUENCY = FREQUENCY
        self.ORIENTATION = ORIENTATION
        self.NB_CELLS = NB_CELLS

        self.PI = 3.141592
        self.CELL_SIZE = 2/ self.NB_CELLS
        self.BANDWITH = 1 / self.CELL_SIZE

    def generate_grid_points(self):
        x_cells = linspace(-1, 1, self.NB_CELLS+1)
        y_cells = linspace(-1, 1, self.NB_CELLS+1)
        points = [[0 for _ in range(self.NB_CELLS)] for _ in range(self.NB_CELLS)]
        for y in range(len(y_cells)-1):
            for x in range(len(x_cells)-1):
                kx = random.uniform(x_cells[x],x_cells[x+1])
                ky = random.uniform(y_cells[y],y_cells[y+1])
                points[self.NB_CELLS-y-1][x] = (kx, ky, (x_cells[x], x_cells[x+1], self.RESOLUTION//self.NB_CELLS * x, self.RESOLUTION//self.NB_CELLS * (x+1)),(y_cells[y], y_cells[y+1], self.RESOLUTION//self.NB_CELLS * y, self.RESOLUTION//self.NB_CELLS * (y+1)))
        return points


@ti.data_oriented
class GaborTaichi(Noise):

    def __init__(self, RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS):
        super().__init__(RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS)

        self.gabor_img = ti.field(dtype=float, shape=(self.RESOLUTION, self.RESOLUTION))


    def calcul_gabor_noise(self):
        positions = self.generate_grid_points()
        for i1,i2 in ti.ndrange(len(positions),len(positions[0])):
            c = positions[i1][i2]
            x, y = c[0], c[1]
            xmin, xmax, pxmin, pxmax = c[2]
            ymin, ymax, pymin, pymax = c[3]

            # en gros on regarde les voisins juste à côté
            for decalage in [-1, 1]:
                if (i2 + decalage) >= 0 and (i2 + decalage) < len(positions[0]):
                    cx = positions[i1][i2 + decalage]
                    xmin = min(xmin, cx[2][0])
                    xmax = max(xmax, cx[2][1])
                    pxmin = min(pxmin, cx[2][2])
                    pxmax = max(pxmax, cx[2][3])
                if (i1 + decalage) >= 0 and (i1 + decalage) < len(positions):
                    cy = positions[i1 + decalage][i2]
                    ymin = min(ymin, cy[3][0])
                    ymax = max(ymax, cy[3][1])
                    pymin = min(pymin, cy[3][2])
                    pymax = max(pymax, cy[3][3])
            hauteur = pymax - pymin
            largeur  = pxmax - pxmin
            self.add_gabor(x, y, xmin, xmax, ymin, ymax, pymin, pxmin, hauteur, largeur)


    @ti.kernel
    def add_gabor(self, x0: float, y0: float, xmin: float, xmax: float, ymin: float, ymax: float, pymin: int, pxmin: int, hauteur: int, largeur: int):
        for i, j in ti.ndrange(hauteur, largeur):
            vx = xmin + j * (xmax - xmin) / (largeur - 1)
            vy = ymin + i * (ymax - ymin) / (hauteur - 1)
            gaussian = ti.exp(-self.PI * self.BANDWITH**2 * ((vx - x0)**2 + (vy - y0)**2))
            sinusoid = ti.sin(self.FREQUENCY * ((vx - x0) * ti.cos(self.ORIENTATION) + (vy - y0) * ti.sin(self.ORIENTATION)))
            self.gabor_img[pymin + i, pxmin + j] += gaussian * sinusoid

    
    def show_gabor_noise(self):
        self.calcul_gabor_noise()
        gui = ti.GUI("Gabor Noise", (self.RESOLUTION, self.RESOLUTION))
        while gui.running:
            gui.set_image(self.gabor_img.to_numpy())
            gui.show()





@ti.data_oriented
class PhasorTaichi(Noise):

    def __init__(self, RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS):
        super().__init__(RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS)

        self.phasor_img = ti.field(dtype=float, shape=(self.RESOLUTION, self.RESOLUTION))

        self.phasor_real = ti.field(dtype=float, shape=(self.RESOLUTION, self.RESOLUTION))
        self.phasor_imag = ti.field(dtype=float, shape=(self.RESOLUTION, self.RESOLUTION))

    @ti.kernel
    def add_phasor(self, x0: float, y0: float, xmin: float, xmax: float, ymin: float, ymax: float, pymin: int, pxmin: int, hauteur: int, largeur: int):
        for i, j in ti.ndrange(hauteur, largeur):
            vx = xmin + j * (xmax - xmin) / (largeur - 1)
            vy = ymin + i * (ymax - ymin) / (hauteur - 1)
            gaussian = ti.exp(-self.PI * self.BANDWITH**2 * ((vx - x0)**2 + (vy - y0)**2))
            cosin = ti.cos(self.FREQUENCY * ((vx - x0) * ti.cos(self.ORIENTATION) + (vy - y0) * ti.sin(self.ORIENTATION)))
            sinus = ti.sin(self.FREQUENCY * ((vx - x0) * ti.cos(self.ORIENTATION) + (vy - y0) * ti.sin(self.ORIENTATION)))
            self.phasor_real[pymin + i, pxmin + j] += gaussian * cosin
            self.phasor_imag[pymin + i, pxmin + j] += gaussian * sinus

    @ti.kernel
    def phasor_final(self):
        for i, j in ti.ndrange(self.RESOLUTION, self.RESOLUTION):
            self.phasor_img[i, j] = ti.sin(ti.atan2(self.phasor_real[i, j], self.phasor_imag[i, j]))

    def calcul_phasor_noise(self):
        positions = self.generate_grid_points()
        for i1,i2 in ti.ndrange(len(positions),len(positions[0])):
            c = positions[i1][i2]
            x, y = c[0], c[1]
            xmin, xmax, pxmin, pxmax = c[2]
            ymin, ymax, pymin, pymax = c[3]
            for decalage in [-1, 1]:
                if (i2 + decalage) >= 0 and (i2 + decalage) < len(positions[0]):
                    cx = positions[i1][i2 + decalage]
                    xmin = min(xmin, cx[2][0])
                    xmax = max(xmax, cx[2][1])
                    pxmin = min(pxmin, cx[2][2])
                    pxmax = max(pxmax, cx[2][3])
                    
                if (i1 + decalage) >= 0 and (i1 + decalage) < len(positions):
                    cy = positions[i1 + decalage][i2]
                    ymin = min(ymin, cy[3][0])
                    ymax = max(ymax, cy[3][1])
                    pymin = min(pymin, cy[3][2])
                    pymax = max(pymax, cy[3][3])
            hauteur = pymax - pymin
            largeur  = pxmax - pxmin
            self.add_phasor(x, y, xmin, xmax, ymin, ymax, pymin, pxmin, hauteur, largeur)
        self.phasor_final()

    def show_phasor_noise(self):
        self.calcul_phasor_noise()
        gui = ti.GUI("Phasor Noise", (self.RESOLUTION, self.RESOLUTION))
        while gui.running:
            gui.set_image(self.phasor_img.to_numpy())
            gui.show()



if __name__ == '__main__':
    g1 = GaborTaichi(1000, 75., 0.5, 10)
    g1.show_gabor_noise()

    p1 = PhasorTaichi(1000, 75., 0.5, 10)
    p1.show_phasor_noise()

