import numpy as np

def lens(u1, x1, y1, f, lmbd=500e-9):
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
    N = u1.shape[0]
    W1 = (x1[0,1]-x1[0,0])*N
    W2 = N*lmbd*f/W1
    delta1 = W1/N
    delta2 = W2/N
    
    # Compute meshgrids
    nx = np.arange(N)
    ny = np.arange(N)
    x2 = -W2/2 + delta2/2 + nx*delta2
    y2 = -W2/2 + delta2/2 + ny*delta2
    nx, ny = np.meshgrid(nx, ny)
    x2, y2 = np.meshgrid(x2, y2)
    c1 = -W1/2 + delta1/2
    c2 = -W2/2 + delta2/2
    
    # Compute fourier transform
    phase_fact = np.exp(-1j*(2*np.pi)*(1/(lmbd*f))*(nx+ny)*delta1*c2)
    u2 = np.fft.fft2(u1*phase_fact)
    u2 = u2*(delta1*delta1/(lmbd*f))*np.exp(-1j*2*np.pi*(x2+y2)*c1*(1/(lmbd*f)))
    
    return u2, x2, y2