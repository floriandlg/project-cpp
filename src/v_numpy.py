import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy.fft import fft2, fftshift    



SEED = 1


rng = default_rng(SEED)



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
        x_cells = np.linspace(-1, 1, self.NB_CELLS, endpoint=False) + self.CELL_SIZE / 2
        y_cells = np.linspace(-1, 1, self.NB_CELLS, endpoint=False) + self.CELL_SIZE / 2
        points = []
        for x in x_cells:
            for y in y_cells:
                x_offset = rng.uniform(-self.CELL_SIZE/2, self.CELL_SIZE/2)
                y_offset = rng.uniform(-self.CELL_SIZE/2, self.CELL_SIZE/2)
                points.append((x + x_offset, y + y_offset))
        return points




class GaborNumpy(Noise):

    def __init__(self, RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS):
        super().__init__(RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS)

    def gabor_kernel(self,x, y, x0, y0):
        gaussian = np.exp(-self.PI * self.BANDWITH**2 * ((x - x0)**2 + (y - y0)**2))
        sinusoid = np.sin(self.FREQUENCY * ((x - x0) * np.cos(self.ORIENTATION) + (y - y0) * np.sin(self.ORIENTATION)))
        return gaussian * sinusoid

    def gabor_noise(self):
        x = np.linspace(-1, 1, self.RESOLUTION)
        y = np.linspace(-1, 1, self.RESOLUTION)
        X, Y = np.meshgrid(x, y)
        positions = self.generate_grid_points()
        gabor_noise = np.zeros((self.RESOLUTION, self.RESOLUTION))
        for x0, y0 in positions:
            gabor_noise += self.gabor_kernel(X, Y, x0, y0)
        return gabor_noise

    def show_gabor_noise(self):
        gabor_noise_show = self.gabor_noise()
        plt.figure(figsize=(6, 6))
        plt.imshow(gabor_noise_show, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
        plt.title(f"Bruit de Gabor sur une grille ({self.NB_CELLS}x{self.NB_CELLS})")
        plt.show()

    def show_gabor_noise_fft(self):
        gabor_noise_fft = fftshift(fft2(self.gabor_noise()))
        magnitude_spectrum = np.abs(gabor_noise_fft)**0.5
        plt.figure(figsize=(6, 6))
        plt.imshow(magnitude_spectrum, interpolation='nearest', extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
        plt.title("Fourier Transform of Gabor Noise")
        plt.xlim(-0.2, 0.2)
        plt.ylim(-0.2, 0.2)
        plt.show()

    def show_gabor_noise_variance(self):
        gabor_noise_fft = fftshift(fft2(self.gabor_noise()**2))
        magnitude_spectrum = np.abs(gabor_noise_fft)**0.5
        plt.figure(figsize=(6, 6))
        plt.imshow(magnitude_spectrum, interpolation='nearest', extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
        plt.title("Fourier Transform of Gabor Noise")
        plt.xlim(-0.2, 0.2)
        plt.ylim(-0.2, 0.2)
        plt.show()




class PhasorNumpy(Noise):
    def __init__(self, RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS):
        super().__init__(RESOLUTION, FREQUENCY, ORIENTATION, NB_CELLS)

    def phasor_noise(self):
        x = np.linspace(-1, 1, self.RESOLUTION)
        y = np.linspace(-1, 1, self.RESOLUTION)
        X, Y = np.meshgrid(x, y)
        positions = self.generate_grid_points()
        real_part = np.zeros((self.RESOLUTION, self.RESOLUTION))
        imag_part = np.zeros((self.RESOLUTION, self.RESOLUTION))
        for x0, y0 in positions:
            gaussian = np.exp(-self.PI * self.BANDWITH**2 * ((X - x0)**2 + (Y - y0)**2))
            cosine_wave = np.cos(self.FREQUENCY * ((X - x0) * np.cos(self.ORIENTATION) + (Y - y0) * np.sin(self.ORIENTATION)))
            sine_wave = np.sin(self.FREQUENCY * ((X - x0) * np.cos(self.ORIENTATION) + (Y - y0) * np.sin(self.ORIENTATION)))
            real_part += gaussian * cosine_wave
            imag_part += gaussian * sine_wave
        phase = np.arctan2(imag_part, real_part)
        return np.sin(phase)

    def show_phasor_noise(self):
        phasor_noise_show = self.phasor_noise()
        plt.figure(figsize=(6, 6))
        plt.imshow(phasor_noise_show, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
        plt.title(f"Bruit de Phasor sur une grille  ({self.NB_CELLS}x{self.NB_CELLS})")
        plt.show()

    def show_phasor_noise_fft(self):
        phasor_noise_fft = fftshift(fft2(self.phasor_noise()))
        magnitude_spectrum = np.abs(phasor_noise_fft)**0.5 
        plt.figure(figsize=(6, 6))
        plt.imshow(magnitude_spectrum, interpolation='nearest', extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
        plt.title("Fourier Transform of Phasor Noise")
        plt.xlim(-0.2, 0.2)
        plt.ylim(-0.2, 0.2)
        plt.show()

    def show_phasor_noise_variance(self):
        phasor_noise_fft = fftshift(fft2(self.phasor_noise()**2))
        magnitude_spectrum = np.abs(phasor_noise_fft)**0.5 
        plt.figure(figsize=(6, 6))
        plt.imshow(magnitude_spectrum, interpolation='nearest', extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
        plt.title("Fourier Transform of Phasor Noise")
        plt.xlim(-0.2, 0.2)
        plt.ylim(-0.2, 0.2)
        plt.show()

"""
def show_comparison_noise():
    gabor_noise_show = gabor_noise()
    phasor_noise_show = phasor_noise()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(gabor_noise_show, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    axs[0].set_title(f"Gabor Noise")

    axs[1].imshow(phasor_noise_show, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    axs[1].set_title(f"Phasor Noise")

    plt.show()

def show_comparison_noise_variance():

    gabor_noise_fft = fftshift(fft2(gabor_noise()**2))
    phasor_noise_fft = fftshift(fft2(phasor_noise()**2))

    magnitude_spectrum_gabor = np.abs(gabor_noise_fft)**0.5
    magnitude_spectrum_phasor = np.abs(phasor_noise_fft)**0.5

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(magnitude_spectrum_gabor, interpolation='nearest', extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    axs[0].set_title("Fourier Transform of Gabor Noise ")
    axs[0].set_xlim(-0.2, 0.2)
    axs[0].set_ylim(-0.2, 0.2)

    axs[1].imshow(magnitude_spectrum_phasor, interpolation='nearest', extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    axs[1].set_title("Fourier Transform of Phasor Noise")
    axs[1].set_xlim(-0.2, 0.2)
    axs[1].set_ylim(-0.2, 0.2)

    plt.show()"""

if __name__ == "__main__":
    g1 = GaborNumpy(1000, 100, 0.5, 10)
    g1.show_gabor_noise()
    

    p1 = PhasorNumpy(1000, 100, 0.5, 10)
    p1.show_phasor_noise()
    
    