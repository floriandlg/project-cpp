import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy.fft import fft2, fftshift

PERIOD = 0.2
ORIENTATION = 0.75

SEED = 42
PI = 3.141592
FREQUENCY = 1/PERIOD
RESOLUTION = 1024

MAGNITUDE = 1
WIDTH = 2
NB_KERNELS = 16

rng = default_rng(SEED)

def gaussian_envelope(x, y, x0, y0, width):
    return np.exp(-PI * width**2 * ((x - x0)**2 + (y - y0)**2))

def sinusoid(x, y, x0, y0, orientation):
    return np.cos(2 * PI * FREQUENCY * ((x - x0) * np.cos(orientation) + (y - y0) * np.sin(orientation)))

def gabor_kernel(x,y,MAGNITUDE,WIDTH):
    return MAGNITUDE*np.exp(-PI*WIDTH**2*(x**2+y**2))*np.cos(2*PI*FREQUENCY*(x*np.cos(ORIENTATION)+y*np.sin(ORIENTATION)))

def show_gaussian_envelope():
    x = np.linspace(-1, 1, RESOLUTION)
    y = np.linspace(-1, 1, RESOLUTION)
    X, Y = np.meshgrid(x, y)
    Z = gaussian_envelope(X, Y, 0, 0, WIDTH)
    plt.imshow(Z, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title("Enveloppe Gaussienne")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def show_sinusoid():
    x = np.linspace(-1, 1, RESOLUTION)
    y = np.linspace(-1, 1, RESOLUTION)
    X, Y = np.meshgrid(x, y)
    Z = sinusoid(X, Y, 0, 0, ORIENTATION)
    plt.imshow(Z, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title("Vague sinusoïdale")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def show_gabor_kernel():
    x = np.linspace(-1,1,RESOLUTION)
    y = np.linspace(-1,1,RESOLUTION)
    X,Y = np.meshgrid(x,y)
    Z = gabor_kernel(X,Y,MAGNITUDE,WIDTH)
    plt.imshow(Z,extent=(-1,1,-1,1),cmap='gray',origin='lower')
    plt.title("Noyau de Gabor")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def show_gabor_kernel_fft():
    x = np.linspace(-1, 1, RESOLUTION)
    y = np.linspace(-1, 1, RESOLUTION)
    X, Y = np.meshgrid(x, y)
    Z = gabor_kernel(X, Y, MAGNITUDE, WIDTH)
    Z_fft = fftshift(fft2(Z))
    plt.imshow(np.abs(Z_fft), extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    plt.title("Spectre de Fourier du noyau de Gabor")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.show()

def show_all_in_one():
    x = np.linspace(-1, 1, RESOLUTION)
    y = np.linspace(-1, 1, RESOLUTION)
    X, Y = np.meshgrid(x, y)

    # Gabor Kernel
    Z_gabor = gabor_kernel(X, Y, MAGNITUDE, WIDTH)

    # Gaussian Envelope
    Z_gaussian = gaussian_envelope(X, Y, 0, 0, WIDTH)

    # Sinusoid
    Z_sinusoid = sinusoid(X, Y, 0, 0, ORIENTATION)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Gabor Kernel
    axes[0].imshow(Z_gabor, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    axes[0].set_title("Noyau de Gabor")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # Plot Gaussian Envelope
    axes[1].imshow(Z_gaussian, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    axes[1].set_title("Enveloppe Gaussienne")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    # Plot Sinusoid
    axes[2].imshow(Z_sinusoid, extent=(-1, 1, -1, 1), cmap='gray', origin='lower')
    axes[2].set_title("Vague Sinusoïdale")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_all_in_one()
