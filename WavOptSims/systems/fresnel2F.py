import numpy as np
from tqdm import tqdm

from utils import lens_fourier, slm_ramp, lens_quadratic, fresnel_prop, fresnel_prop_ir, fresnel_prop_tf

class fresnel2F:
    """
    2F system
    <->lens<->

    Attributes:
    f : float
    Focal length of all lenses
    """

    def __init__(self, f, lmbd=500e-9):
        self.f = f
        self.lmbd = lmbd

    def forward(self, u1, x1, y1):
        """
        Function to pass input field through
        2F system

        Parameters:
        u1 : np.nddarray
        Input field
        x1 : np.ndarray
        x coords of input field, uniformly distributed
        y1 : np.ndarray
        y coords of input field, uniformly distributed
        """

        # Propagate to lens
        u2, x2, y2 = fresnel_prop(u1, x1, y1, 2*self.f, self.lmbd)

        # lens
        u3, x3, y3 = lens_quadratic(u2, x2, y2, self.f, self.lmbd)

        # Propagate to focus plane
        u4, x4, y4 = fresnel_prop(u3, x3, y3, 2*self.f, self.lmbd)

        return u4, x4, y4

    def run(self, u1, x1, y1, args):
        """
        Run system multiple times with random input phase
        to mitigate interference effects

        u1 : np.nddarray
        Input field
        x1 : np.ndarray
        x coords of input field, uniformly distributed
        y1 : np.ndarray
        y coords of input field, uniformly distributed

        Return:
        im_out : np.ndarray
        Magnitude of output field
        x_out : np.ndarray
        x coords of output field, uniformly distributed
        y_out : np.ndarray
        y coords of output field, uniformly distributed
        """

        im_out = np.zeros_like(u1, dtype=np.float64)

        for i in tqdm(range(args.iters)):
            u_out, x_out, y_out = self.forward(u1, x1, y1)
            im_out += np.abs(u_out)**2

        im_out = im_out/args.iters

        return im_out, x_out, y_out