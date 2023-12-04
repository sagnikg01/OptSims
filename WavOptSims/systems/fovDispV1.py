import numpy as np

from utils import lens_fourier, slm_ramp, lens_quadratic


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
    """

    def __init__(self, f, P, lmbd):
        self.f = f
        self.P = P
        self.lmbd = lmbd

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

        # Quadratic Phase
        u3, x3, y3 = lens_quadratic(u2, x2, y2, self.P, self.lmbd)

        # Lens 2
        u4, x4, y4 = lens_fourier(u3, x3, y3, self.f, self.lmbd)

        # SLM Phase Ramp
        u5, x5, y5 = slm_ramp(u4, x4, y4, vx, vy, self.lmbd)

        # Lens 3
        u6, x6, y6 = lens_fourier(u5, x5, y5, self.f, self.lmbd)

        # Quadratic Phase
        u7, x7, y7 = lens_quadratic(u6, x6, y6, -self.P, self.lmbd)

        # Lens 4
        u8, x8, y8 = lens_fourier(u7, x7, y7, self.f, self.lmbd)

        if debug:
            U = {"before_slm": u4,
                 "output": u8}
            X = {"before_slm": x4,
                 "output": x8}
            Y = {"before_slm": y4,
                 "output": y8}

            return U, X, Y

        return u8, x8, y8
