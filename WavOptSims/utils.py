import numpy as np


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