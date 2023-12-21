import numpy as np
from tqdm import tqdm

from utils import lens_fourier, slm_ramp, lens_quadratic, fresnel_prop


class fovDispV1:
    """
    Foveated display system with 2 4f systems
    and a phase SLM
    <-f->lens<-f->quad_phase<-f->lens<-f->SLM<-f->lens \
    <-f->quad_phase<-f->lens

    Attributes:
    f : float
    Focal length of all lenses
    P : float
    Quadratic phase power
    aprtr_ln : float
    Aperture length
    f_ep : float
    Eyepiece focal length
    eye_dist : float
    Distance b/w eyepiece and eye
    f_e : float
    Eye focal length
    """

    def __init__(self, args):
        self.f = args.f
        self.P = args.P
        self.lmbd = args.lmbd
        self.aprtr_ln = args.aprtr_ln
        self.f_ep = args.f_ep
        self.eye_dist = args.eye_dist
        self.f_e = args.f_e

    def forward(self, u1, x1, y1, vx, vy, debug=False):
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
        vx : float
        slope of ramp in x-direction
        vy : float
        slope of ramp in y-direction

        Return:
        U : dict
        intermediate fields un
        X : dict
        intermediate x coord fields xn
        Y : dict
        intermediate y coord fields yn
        un : np.nddarray
        n-th output field
        xn : np.ndarray
        x coords of n-th output field, uniformly distributed
        yn : np.ndarray
        y coords of n-th output field, uniformly distributed
        """

        # Lens 1
        u2, x2, y2 = lens_fourier(u1, x1, y1, self.f, self.lmbd)
        u2[x2**2+y2**2 > self.aprtr_ln] = 0

        # Quadratic Phase
        u3, x3, y3 = lens_quadratic(u2, x2, y2, 1/self.P, self.lmbd)
        u3[x3**2+y3**2 > self.aprtr_ln] = 0

        # Lens 2
        u4, x4, y4 = lens_fourier(u3, x3, y3, self.f, self.lmbd)
        u4[x4**2+y4**2 > self.aprtr_ln] = 0

        # SLM Phase Ramp
        u5, x5, y5 = slm_ramp(u4, x4, y4, vx, vy, self.lmbd)
        u5[x5**2+y5**2 > self.aprtr_ln] = 0

        # Lens 3
        u6, x6, y6 = lens_fourier(u5, x5, y5, self.f, self.lmbd)
        u6[x6**2+y6**2 > self.aprtr_ln] = 0

        # Quadratic Phase
        u7, x7, y7 = lens_quadratic(u6, x6, y6, -1/self.P, self.lmbd)
        u7[x7**2+y7**2 > self.aprtr_ln] = 0

        # Lens 4
        u8, x8, y8 = lens_fourier(u7, x7, y7, self.f, self.lmbd)

        # Propagate to eyepiece
        u9, x9, y9 = fresnel_prop(u8, x8, y8, self.f_ep, self.lmbd)

        # Eyepice
        u10, x10, y10 = lens_quadratic(u9, x9, y9, self.f_ep, self.lmbd)

        # Propagate to eye
        u11, x11, y11 = fresnel_prop(u10, x10, y10, self.eye_dist, self.lmbd)

        # Eye lens
        u12, x12, y12 = lens_quadratic(u11, x11, y11, self.f_e, self.lmbd)

        # Propagate to eye
        u13, x13, y13 = fresnel_prop(u12, x12, y12, self.f_e, self.lmbd)

        if debug:
            U = {"before_slm": u4,
                 "output": u8,
                 "eye": u13}
            X = {"before_slm": x4,
                 "output": x8,
                 "eye": x13}
            Y = {"before_slm": y4,
                 "output": y8,
                 "eye": y13}

            return U, X, Y

        return u8, x8, y8
    
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
            u_out, x_out, y_out = self.forward(u1, x1, y1, args.vx, args.vy)
            im_out += np.abs(u_out)**2

        im_out = im_out/args.iters

        return im_out, x_out, y_out
