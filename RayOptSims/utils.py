import numpy as np

def lens(h1, theta1, f):
    """
    Function to transform a light ray using lens

    Parameters:
    h1 : float
    input height from axis
    theta1 : float
    input angle wrt axis
    f : float
    focal length of lens

    Returns:
    h2 : float
    output height from axis
    theta2 : float
    output angle wrt axis
    """

    h2 = h1
    theta2 = theta1 - h1/f

    return h2, theta2

def free_space(h1, theta1, d):
    """
    Function to simulate free space propagation

    Parameters:
    h1 : float
    input height from axis
    theta1 : float
    input angle wrt axis
    d : float
    distance of propagation

    Returns:
    h2 : float
    output height from axis
    theta2 : float
    output angle wrt axis
    """
    
    h2 = h1 + d*theta1
    theta2 = theta1

    return h2, theta2

def slm_tilt(h1, theta1, tilt):
    """
    Function to simulate SLM phase ramp

    Parameters:
    h1 : float
    input height from axis
    theta1 : float
    input angle wrt axis
    tilt : float
    tilt of ramp in rad

    Returns:
    h2 : float
    output height from axis
    theta2 : float
    output angle wrt axis
    """

    h2 = h1
    theta2 = theta1 + tilt

    return h2, theta2