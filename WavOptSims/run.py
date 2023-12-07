import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import argparse

from systems.fovDispV1 import fovDispV1


def run_system(u1, x1, y1, sys, args):
    """
    Run system multiple times with random input phase
    to mitigate interference effects

    u1 : np.nddarray
    Input field
    x1 : np.ndarray
    x coords of input field, uniformly distributed
    y1 : np.ndarray
    y coords of input field, uniformly distributed
    sys : optical system object

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
        u_out, x_out, y_out = sys.forward(u1, x1, y1, args.vx, args.vy)
        im_out += np.abs(u_out)**2

    im_out = im_out/args.iters

    return im_out, x_out, y_out


def parse_args():

    parser = argparse.ArgumentParser(description='Arguments for FovDispV1')

    parser.add_argument('--impath', type=str,
                        default='data/cameraman.jpg', help='Input image path')
    parser.add_argument('--fovsize', type=int, default=512,
                        help='Fovea size in pixels')
    parser.add_argument('--N', type=int, default=2000,
                        help='Number of samples')
    parser.add_argument('--W1', type=float, default=10e-3,
                        help='Width of input in meters')
    parser.add_argument('--f', type=float, default=100e-3, help='Focal length')
    parser.add_argument('--P', type=float, default=1/(50e-3),
                        help='Power of quadratic phase')
    parser.add_argument('--lmbd', type=float,
                        default=500e-9, help='Wavelength')
    parser.add_argument('--vx', type=float, default=0,
                        help='x-slope of phase ramp')
    parser.add_argument('--vy', type=float, default=0,
                        help='y-slope of phase ramp')
    parser.add_argument('--iters', type=int, default=10,
                        help='Number of iters')
    parser.add_argument('--aprtr_ln', type=float,
                        default=12.5e-3, help='Aperture length')
    parser.add_argument('--f_ep', type=float,
                        default=35e-3, help='eyepiece focal length')
    parser.add_argument('--eye_dist', type=float,
                        default=25e-3, help='dist b/w eyepeice & eye')
    parser.add_argument('--f_e', type=float,
                        default=25e-3, help='eye focal length')

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # Command line args
    args = parse_args()

    # Create input field
    u1 = plt.imread(args.impath)[:, :, 0]/255
    u1 = cv2.resize(u1, (args.fovsize, args.fovsize))
    u1 = np.pad(u1, (args.N-args.fovsize)//2)
    u1 = np.sqrt(u1)
    nx = np.arange(args.N)
    ny = np.arange(args.N)
    nx, ny = np.meshgrid(nx, ny)
    x1 = -args.W1/2 + args.W1/(2*args.N) + args.W1*nx/args.N
    y1 = -args.W1/2 + args.W1/(2*args.N) + args.W1*ny/args.N

    # Create fovDisp system
    fovDispSys = fovDispV1(args.f, args.P, args.aprtr_ln, args.f_ep, args.eye_dist, args.f_e, args.lmbd)

    # Run input through fovDisp
    im_out, x_out, y_out = run_system(u1, x1, y1, fovDispSys, args)

    # Visualize output
    plt.imshow(im_out, cmap="gray")
    plt.show()
    # plt.imsave("out1.png", im_out)
    # print(np.mean(im_out))
    # print(np.mean(np.abs(u1)**2))

