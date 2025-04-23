import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy.fft import fft2, fftshift    
import time

# Settings
RESOLUTION = 1024 # Change resolution of the image
FREQUENCY = 100. # Change frequency of the sinusoid
ORIENTATION = np.pi/4 # Change orientation of the sinusoid
RANDOM_ORIENTATION = (0,np.pi) #à modifier en fonction du résultat voulu
NB_CELLS = 10 # Change cell size
ORIENTATIONS = [0, np.pi/4, np.pi/2, 3*np.pi/4] #à modifier en fonction du résultat voulu

# Constants
PI = 3.141592
TWO_PI = 6.283185
CELL_SIZE = 2/NB_CELLS
BANDWITH = 1.0 / CELL_SIZE
SEED = 1

# Profiles
SINE_PROFILE = False
SQUARE_PROFILE = False
TRIANGULARE_PROFILE = False
SAWTOOTH_PROFILE = True

rng = default_rng(SEED)

def generate_grid_points():
    x_cells = np.linspace(-1, 1, NB_CELLS, endpoint=False) + CELL_SIZE / 2
    y_cells = np.linspace(-1, 1, NB_CELLS, endpoint=False) + CELL_SIZE / 2
    points = []
    for x in x_cells:
        for y in y_cells:
            x_offset = rng.uniform(-CELL_SIZE/2, CELL_SIZE/2)
            y_offset = rng.uniform(-CELL_SIZE/2, CELL_SIZE/2)
            points.append((x + x_offset, y + y_offset))
    return points

def apply_profile(value):
    if SINE_PROFILE:
        return np.sin(value)
    elif SQUARE_PROFILE:
        return np.sign(np.sin(value))
    elif TRIANGULARE_PROFILE:
        return 2 * np.abs(2 * (value / TWO_PI - np.floor(value / TWO_PI + 0.5))) - 1
    elif SAWTOOTH_PROFILE:
        return 2 * (value / TWO_PI - np.floor(value / TWO_PI + 0.5))
    else:
        return value

def gabor_kernel(x, y, x0, y0):
    gaussian = np.exp(-PI * BANDWITH**2 * ((x - x0)**2 + (y - y0)**2))
    sinusoid = np.sin(FREQUENCY * ((x - x0) * np.cos(ORIENTATION) + (y - y0) * np.sin(ORIENTATION)))
    return gaussian * sinusoid

def gabor_noise():
    x = np.linspace(-1, 1, RESOLUTION)
    y = np.linspace(-1, 1, RESOLUTION)
    X, Y = np.meshgrid(x, y)
    positions = generate_grid_points()
    gabor_noise = np.zeros((RESOLUTION, RESOLUTION))
    for x0, y0 in positions:
        gabor_noise += gabor_kernel(X, Y, x0, y0)
    return gabor_noise


def phasor_noise():
    x = np.linspace(-1, 1, RESOLUTION)
    y = np.linspace(-1, 1, RESOLUTION)
    X, Y = np.meshgrid(x, y)
    positions = generate_grid_points()
    real_part = np.zeros((RESOLUTION, RESOLUTION))
    imag_part = np.zeros((RESOLUTION, RESOLUTION))
    for x0, y0 in positions:
        gaussian = np.exp(-PI * BANDWITH**2 * ((X - x0)**2 + (Y - y0)**2))
        #theta = rng.uniform(*RANDOM_ORIENTATION)  #motifs continus sur les spectre (intervalle)
        theta = rng.choice(ORIENTATIONS)       #motifs continus sur les spectres (liste)
        phase2 = FREQUENCY * ((X - x0) * np.cos(theta) + (Y - y0) * np.sin(theta))
        real_part += gaussian * np.cos(phase2)
        imag_part += gaussian * np.sin(phase2)
    phase = np.arctan2(imag_part, real_part)
    return apply_profile(phase)


def show_gabor_noise():
    gabor_noise_show = gabor_noise()
    plt.figure(figsize=(6, 6))
    plt.imshow(gabor_noise_show, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title(f"Bruit de Gabor sur une grille ({NB_CELLS}x{NB_CELLS})")
    plt.show()


def show_phasor_noise():
    time_start = time.time()
    phasor_noise_show = phasor_noise()
    #phasor_noise_show = phasor_noise_multiple_orientations()
    print("Time taken: ", time.time() - time_start)
    plt.figure(figsize=(6, 6))
    plt.imshow(phasor_noise_show, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title(f"Bruit de Phasor sur une grille  ({NB_CELLS}x{NB_CELLS})")
    plt.show()

def show_gabor_noise_fft():
    gabor_noise_fft = fftshift(fft2(gabor_noise()))
    magnitude_spectrum = np.abs(gabor_noise_fft)**0.5
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, interpolation='nearest', extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title("Fourier Transform of Gabor Noise")
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.show()

def show_gabor_noise_variance():
    gabor_noise_fft = fftshift(fft2(gabor_noise()**2))
    magnitude_spectrum = np.abs(gabor_noise_fft)**0.5
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, interpolation='nearest', extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title("Fourier Transform of Gabor Noise")
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.show()

def show_phasor_noise_fft():
    phasor_noise_fft = fftshift(fft2(phasor_noise()))
    magnitude_spectrum = np.abs(phasor_noise_fft)**0.5 
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum, interpolation='nearest', extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title("Fourier Transform of Phasor Noise")
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.show()


def show_phasor_noise_variance():
    phasor_noise_fft = fftshift(fft2(phasor_noise()**2))
    magnitude_spectrum = np.abs(phasor_noise_fft)
    magnitude_spectrum_clipped = np.clip(magnitude_spectrum, 0, 20000)  # Clip values to [0, 1] for better visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude_spectrum_clipped, interpolation='nearest', extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title("Fourier Transform of Phasor Noise")
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.show()

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

    plt.show()

if __name__ == "__main__":
    show_phasor_noise()
    #show_phasor_variance()
    show_phasor_noise_fft()
