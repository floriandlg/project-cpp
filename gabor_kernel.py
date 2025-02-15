import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy.fft import fft2, fftshift

PERIOD = 0.3
ORIENTATION = 0.75

SEED = 42
PI = 3.141592
FREQUENCY = 1/PERIOD
RESOLUTION = 1024

MAGNITUDE = 1
WIDTH = 2
NB_KERNELS = 16

rng = default_rng(SEED)

def gabor_kernel(x,y,MAGNITUDE,WIDTH):
    return MAGNITUDE*np.exp(-PI*WIDTH**2*(x**2+y**2))*np.cos(2*PI*FREQUENCY*(x*np.cos(ORIENTATION)+y*np.sin(ORIENTATION)))

def show_gabor_kernel():
    x = np.linspace(-1,1,RESOLUTION)
    y = np.linspace(-1,1,RESOLUTION)
    X,Y = np.meshgrid(x,y)
    Z = gabor_kernel(X,Y,MAGNITUDE,WIDTH)
    plt.imshow(Z,extent=(-1,1,-1,1),cmap='gray',origin='lower')
    plt.title("Gabor Kernel")
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
    plt.title("Fourier Transform of Gabor Kernel")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    plt.xlim(-0.2, 0.2)
    plt.ylim(-0.2, 0.2)
    plt.show()

if __name__ == "__main__":
    show_gabor_kernel()