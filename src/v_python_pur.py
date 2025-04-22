import matplotlib.pyplot as plt
import math
import random


def linspace(a,b,N):
    return [a + (b-a)/(N-1)*i for i in range(N)]


def zero_array(xn,yn):
    return [[0 for _ in range(xn)] for _ in range(yn)]




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

    




class GaborPur(Noise):

    def __init__(self, RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS):
        super().__init__(RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS)


    def gabor_kernel_array(self,xa, ya, x0, y0):
        out = zero_array(len(xa),len(ya))
        for iy,vy in enumerate(ya):
            for ix,vx in enumerate(xa):
                gaussian = math.exp(-self.PI * self.BANDWITH**2 * ((vx - x0)**2 + (vy - y0)**2))
                sinusoid = math.sin(self.FREQUENCY * ((vx - x0) * math.cos(self.ORIENTATION) + (vy - y0) * math.sin(self.ORIENTATION)))
                out[iy][ix] = gaussian * sinusoid
        return out


    def gabor_noise(self):
        out = zero_array(self.RESOLUTION,self.RESOLUTION)
        positions = self.generate_grid_points()
        for i1 in range(len(positions)):
            for i2 in range(len(positions[0])):
                c = positions[i1][i2]

                x, y = c[0], c[1]
                xmin, xmax, pxmin, pxmax = c[2][0], c[2][1], c[2][2], c[2][3]
                ymin, ymax, pymin, pymax = c[3][0], c[3][1], c[3][2], c[3][3]

                for plop in [-1,1]:
                    if i2 + plop in range(len(positions[0])):
                        cx = positions[i1][i2 + plop]
                        xmin = min(xmin, cx[2][0])
                        xmax = max(xmax, cx[2][1])
                        pxmin = min(pxmin, cx[2][2])
                        pxmax = max(pxmax, cx[2][3])
                    
                    if i1 + plop in range(len(positions)):
                        cy = positions[i1 + plop][i2]
                        ymin = min(ymin, cy[3][0])
                        ymax = max(ymax, cy[3][1])
                        pymin = min(pymin, cy[3][2])
                        pymax = max(pymax, cy[3][3])

                gab = self.gabor_kernel_array(linspace(xmin,xmax,pxmax-pxmin),linspace(ymin,ymax,pymax-pymin), x, y)

                for i in range(len(gab)):
                    for u in range(len(gab[0])):
                        out[pymin+i][pxmin+u] += gab[i][u]
        return out 


    def show_noise(self):
        gabor_noise_show = self.gabor_noise()
        plt.figure(figsize=(6, 6))
        plt.imshow(gabor_noise_show, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
        plt.title(f"gabor")
        plt.show()




class PhasorPur(Noise):

    def __init__(self, RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS):
        super().__init__(RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS)


    def phasor_kernel_array(self,xa,ya,x0,y0):
        real = zero_array(len(xa),len(ya))
        imag = zero_array(len(xa),len(ya))
        for iy,vy in enumerate(ya):
            for ix,vx in enumerate(xa):
                gaussian = math.exp(-self.PI * self.BANDWITH **2 * ((vx - x0)**2 + (vy - y0)**2))
                cosin = math.cos(self.FREQUENCY * ((vx - x0) * math.cos(self.ORIENTATION) + (vy - y0) * math.sin(self.ORIENTATION)))
                sinus = math.sin(self.FREQUENCY * ((vx - x0) * math.cos(self.ORIENTATION) + (vy - y0) * math.sin(self.ORIENTATION)))
                real[iy][ix] += gaussian * cosin
                imag[iy][ix] += gaussian * sinus
        return real,imag


    def phasor_noise(self):
        real = zero_array(self.RESOLUTION,self.RESOLUTION)
        imag = zero_array(self.RESOLUTION,self.RESOLUTION)
        positions = self.generate_grid_points()
        for i1 in range(len(positions)):
            for i2 in range(len(positions[0])):
                c = positions[i1][i2]

                x, y = c[0], c[1]
                xmin, xmax, pxmin, pxmax = c[2][0], c[2][1], c[2][2], c[2][3]
                ymin, ymax, pymin, pymax = c[3][0], c[3][1], c[3][2], c[3][3]

                for plop in [-1,1]:
                    if i2 + plop in range(len(positions[0])):
                        cx = positions[i1][i2 + plop]
                        xmin = min(xmin, cx[2][0])
                        xmax = max(xmax, cx[2][1])
                        pxmin = min(pxmin, cx[2][2])
                        pxmax = max(pxmax, cx[2][3])
                    
                    if i1 + plop in range(len(positions)):
                        cy = positions[i1 + plop][i2]
                        ymin = min(ymin, cy[3][0])
                        ymax = max(ymax, cy[3][1])
                        pymin = min(pymin, cy[3][2])
                        pymax = max(pymax, cy[3][3])

                pha1, pha2 = self.phasor_kernel_array(linspace(xmin,xmax,pxmax-pxmin),linspace(ymin,ymax,pymax-pymin), x, y)

                
                for i in range(len(pha1)):
                    for u in range(len(pha1[0])):
                        real[pymin+i][pxmin+u] += pha1[i][u]
                
                for i in range(len(pha2)):
                    for u in range(len(pha2[0])):
                        imag[pymin+i][pxmin+u] += pha2[i][u]
        
        for i in range(len(real)):
            for u in range(len(real[0])):
                real[i][u] = math.sin(math.atan2(real[i][u], imag[i][u]))
               
        return real 


    def show_noise(self):
        gabor_noise_show = self.phasor_noise()
        plt.figure(figsize=(6, 6))
        plt.imshow(gabor_noise_show, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
        plt.title(f"phasor")
        plt.show()







if '__main__' in __name__:
    p1 = PhasorPur(250, 100, 0.5, 10)
    p1.show_noise()

    

