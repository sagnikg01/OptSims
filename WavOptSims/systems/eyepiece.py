import numpy as np
from tqdm import tqdm

from utils import lens_fourier, slm_ramp, lens_quadratic, fresnel_prop, fresnel_prop_ir, fresnel_prop_tf

class Eyepiece:
    """
    Eyepiece simulation

    Attributes:
    f_ep : float
    Eyepiece focal length
    eye_dist : float
    Distance b/w eyepiece and eye
    f_e : float
    Eye focal length
    """

    def __init__(self, args):
        self.lmbd = args.lmbd
        self.f_ep = args.f_ep
        self.eye_dist = args.eye_dist
        self.f_e = args.f_e

    def forward(self, u1, x1, y1, debug=False):
        """
        Function to pass input field through
        FovDisp system

        Parameters:
        u1 : np.nddarray
        Input field
        x1 : np.ndarray
        x coords of input field, uniformly distributed
        y1 : np.ndarray
        y coords of input field, uniformly distributed

        Return:
        un : np.nddarray
        n-th output field
        xn : np.ndarray
        x coords of n-th output field, uniformly distributed
        yn : np.ndarray
        y coords of n-th output field, uniformly distributed
        """

        # Propagate to eyepiece
        u2, x2, y2 = fresnel_prop_tf(u1, x1, y1, self.f_ep, self.lmbd)

        # Eyepice
        u3, x3, y3 = lens_quadratic(u2, x2, y2, self.f_ep, self.lmbd)

        # Propagate to eye
        u4, x4, y4 = fresnel_prop_tf(u3, x3, y3, self.eye_dist, self.lmbd)

        # Eye lens
        u5, x5, y5 = lens_quadratic(u4, x4, y4, self.f_e, self.lmbd)

        # Propagate to eye
        u6, x6, y6 = fresnel_prop_tf(u5, x5, y5, self.f_e, self.lmbd)

        return u6, x6, y6
    
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

