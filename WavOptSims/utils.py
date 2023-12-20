import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import argparse


def lens_fourier(u1, x1, y1, f, lmbd=500e-9):
    """
    Function to transform field from input plane-1
    to focal plane-2

    Parameters:
    u1 : np.nddarray
    Input field
    x1 : np.ndarray
    x coords of input field, uniformly distributed
    y1 : np.ndarray
    y coords of input field, uniformly distributed
    f : float
    focal length

    Returns:
    u2 : np.nddarray
    Output field
    x2 : np.ndarray
    x coords of output field, uniformly distributed
    y2 : np.ndarray
    y coords of output field, uniformly distributed
    """

    # Intial params
    N = u1.shape[0]  # Number of samples
    W1 = (x1[0, 1]-x1[0, 0])*N  # Input width
    W2 = N*lmbd*f/W1  # Output Width
    delta1 = W1/N  # Input spacing
    delta2 = W2/N  # Output Spacing
    c1 = -W1/2 + delta1/2  # Bias 1
    c2 = -W2/2 + delta2/2  # Bias 2

    # Compute meshgrids
    nx = np.arange(N)
    ny = np.arange(N)
    x2 = -W2/2 + delta2/2 + nx*delta2
    y2 = -W2/2 + delta2/2 + ny*delta2
    nx, ny = np.meshgrid(nx, ny)
    x2, y2 = np.meshgrid(x2, y2)

    # Compute fourier transform
    phase_fact = np.exp(-1j*(2*np.pi)*(1/(lmbd*f))*(nx+ny)*delta1*c2)
    u2 = np.fft.fft2(u1*phase_fact)
    u2 = u2*(delta1*delta1/(lmbd*f)) * \
        np.exp(-1j*2*np.pi*(x2+y2)*c1*(1/(lmbd*f)))

    return u2, x2, y2


def slm_ramp(u1, x1, y1, vx, vy, lmbd=500e-9):
    """
    Function to simulate SLM with linear ramp

    Parameters:
    u1 : np.nddarray
    Input field
    x1 : np.ndarray
    x coords of input field, uniformly distributed
    y1 : np.ndarray
    y coords of input field, uniformly distributed
    vx : float
    slope of ramp in x-direction
    vy : float
    slope of ramp in y-direction

    Returns:
    u2 : np.nddarray
    Output field
    x2 : np.ndarray
    x coords of output field, uniformly distributed
    y2 : np.ndarray
    y coords of output field, uniformly distributed
    """

    # Multiply by phase ramp
    ramp = np.exp(-1j*2*(np.pi/lmbd)*(x1*vx+y1*vy))
    u2 = u1*ramp

    # x and y coords
    x2 = x1
    y2 = y1

    return u2, x2, y2


def lens_quadratic(u1, x1, y1, P, lmbd=500e-9):
    """
    Function to simulate lens as a quadratic phase shift
    in the same plane

    Parameters:
    u1 : np.nddarray
    Input field
    x1 : np.ndarray
    x coords of input field, uniformly distributed
    y1 : np.ndarray
    y coords of input field, uniformly distributed
    P : float
    power of lens

    Returns:
    u2 : np.nddarray
    Output field
    x2 : np.ndarray
    x coords of output field, uniformly distributed
    y2 : np.ndarray
    y coords of output field, uniformly distributed
    """

    # Multiply by quadratic phase
    quad_phase = np.exp(-1j*(np.pi*P/lmbd)*(x1**2+y1**2))
    u2 = u1*quad_phase

    # x and y coords
    x2 = x1
    y2 = y1

    return u2, x2, y2

def fresnel_prop_tf(u1, x1, y1, z, lmbd=500e-9):
    """
    Function to simulate fresnel propagation
    using transfer function method

    Parameters:
    u1 : np.nddarray
    Input field
    x1 : np.ndarray
    x coords of input field, uniformly distributed
    y1 : np.ndarray
    y coords of input field, uniformly distributed
    z : float
    propagation distance

    Returns:
    u2 : np.nddarray
    Output field
    x2 : np.ndarray
    x coords of output field, uniformly distributed
    y2 : np.ndarray
    y coords of output field, uniformly distributed
    """

    L = np.max(x1)-np.min(x1) # Side length
    N = u1.shape[0] # Num samples
    dx = L/N # Sample Interval
    k = 2*np.pi/lmbd # Wavenumber

    # Frequency Coords
    #fx = np.arange(-1/(2*dx),1/(2*dx),1/L)
    fx = np.linspace(-1/(2*dx),1/(2*dx)-1/L,N)
    FX, FY = np.meshgrid(fx,fx)

    # Transfer function
    H = np.exp(-1j*np.pi*lmbd*z*(FX**2+FY**2))
    H = np.fft.fftshift(H)

    # Apply TF
    U1 = np.fft.fft2(np.fft.fftshift(u1))
    U2 = H*U1
    u2 = np.fft.ifftshift(np.fft.ifft2(U2))

    return u2, x1, y1

def fresnel_prop_ir(u1, x1, y1, z, lmbd=500e-9):
    """
    Function to simulate fresnel propagation
    using impulse response method

    Parameters:
    u1 : np.nddarray
    Input field
    x1 : np.ndarray
    x coords of input field, uniformly distributed
    y1 : np.ndarray
    y coords of input field, uniformly distributed
    z : float
    propagation distance

    Returns:
    u2 : np.nddarray
    Output field
    x2 : np.ndarray
    x coords of output field, uniformly distributed
    y2 : np.ndarray
    y coords of output field, uniformly distributed
    """

    L = np.max(x1)-np.min(x1) # Side length
    dx = L/u1.shape[0] # Sample Interval
    k = 2*np.pi/lmbd # Wavenumber

    # Spatial coords
    x = np.arange(-L/2,L/2,dx)
    X, Y = np.meshgrid(x,x)

    # Impulse response
    h = 1/(1j*lmbd*z)*np.exp(1j*k/(2*z)*(X**2+Y**2))
    H = np.fft.fft2(np.fft.fftshift(h))*dx**2
    U1 = np.fft.fft2(np.fft.fftshift(u1))
    U2 = H*U1
    u2 = np.fft.ifftshift(np.fft.ifft2(U2))

    return u2, x1, y1

def fresnel_prop(u1, x1, y1, z, lmbd=500e-9):
    """
    Function to simulate fresnel propagation

    Parameters:
    u1 : np.nddarray
    Input field
    x1 : np.ndarray
    x coords of input field, uniformly distributed
    y1 : np.ndarray
    y coords of input field, uniformly distributed
    z : float
    propagation distance

    Returns:
    u2 : np.nddarray
    Output field
    x2 : np.ndarray
    x coords of output field, uniformly distributed
    y2 : np.ndarray
    y coords of output field, uniformly distributed
    """

    L = np.max(x1)-np.min(x1) # Side length
    dx = L/u1.shape[0] # Sample Interval

    flag = (dx >= lmbd*z/L)

    if not flag:
        return fresnel_prop_tf(u1, x1, y1, z, lmbd=500e-9)
    else:
        return fresnel_prop_ir(u1, x1, y1, z, lmbd=500e-9)
    

if __name__ == "__main__":

    ### Test lens_fourier func ###
    pass
    ### Test slm_ramp func ###
    pass
    ### Test lens_quadratic func ###
    pass

