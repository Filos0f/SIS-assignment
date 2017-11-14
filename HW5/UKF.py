from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from filterpy.common import dot3
from filterpy.kalman import unscented_transform
from filterpy.stats import logpdf
import math
import numpy as np
from numpy import eye, zeros, dot, isscalar, outer
from scipy.linalg import inv, cholesky
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from GetRadar import GetRadar
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise



def fx(x, dt):
    """ state transition function for sstate [downrange, vel, altitude]"""
    F = np.array([[1., dt, 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])

    return np.dot(F, x)


def hx(x):
    """ returns slant range based on downrange distance and altitude"""

    return (x[0]**2 + x[2]**2)**.5

class UnscentedKalmanFilter(object):
    def __init__(self, dim_x, dim_z, dt, hx, fx, points,
                 sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
                 residual_x=None,
                 residual_z=None):

        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn
        self.log_likelihood = 0.0

        if sqrt_fn is None:
            self.msqrt = cholesky
        else:
            self.msqrt = sqrt_fn

        # weights for the means and covariances.
        self.Wm, self.Wc = self.points_fn.weights()

        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update

        self.sigmas_f = zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = zeros((self._num_sigmas, self._dim_z))


    def predict(self, dt=None,  UT=None, fx_args=()):
        if dt is None:
            dt = self._dt

        if not isinstance(fx_args, tuple):
            fx_args = (fx_args,)

        if UT is None:
            UT = unscented_transform

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        for i in range(self._num_sigmas):
            self.sigmas_f[i] = self.fx(sigmas[i], dt, *fx_args)

        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q,
                            self.x_mean, self.residual_x)


    def update(self, z, R=None, UT=None, hx_args=()):
        if z is None:
            return

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if UT is None:
            UT = unscented_transform

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R

        for i in range(self._num_sigmas):
            self.sigmas_h[i] = self.hx(self.sigmas_f[i], *hx_args)

        # mean and covariance of prediction passed through unscented transform
        zp, Pz = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)

        # compute cross variance of the state and the measurements
        Pxz = zeros((self._dim_x, self._dim_z))
        for i in range(self._num_sigmas):
            dx = self.residual_x(self.sigmas_f[i], self.x)
            dz =  self.residual_z(self.sigmas_h[i], zp)
            Pxz += self.Wc[i] * outer(dx, dz)


        self.K = dot(Pxz, inv(Pz))        # Kalman gain
        self.y = self.residual_z(z, zp)   # residual

        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot3(self.K, Pz, self.K.T)

        self.log_likelihood = logpdf(self.y, np.zeros(len(self.y)), Pz)


    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        Pxz = zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = self.residual_x(sigmas_f[i], x)
            dz =  self.residual_z(sigmas_h[i], z)
            Pxz += self.Wc[i] * outer(dx, dz)


    @property
    def likelihood(self):
        return math.exp(self.log_likelihood)


    def batch_filter(self, zs, Rs=None, UT=None):

        try:
            z = zs[0]
        except:
            assert not isscalar(zs), 'zs must be list-like'

        if self._dim_z == 1:
            assert isscalar(z) or (z.ndim==1 and len(z) == 1), \
            'zs must be a list of scalars or 1D, 1 element arrays'

        else:
            assert len(z) == self._dim_z, 'each element in zs must be a' \
            '1D array of length {}'.format(self._dim_z)

        z_n = np.size(zs, 0)
        if Rs is None:
            Rs = [None] * z_n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = zeros((z_n, self._dim_x))
        else:
            means = zeros((z_n, self._dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((z_n, self._dim_x, self._dim_x))

        for i, (z, r) in enumerate(zip(zs, Rs)):
            self.predict(UT=UT)
            self.update(z, r, UT=UT)
            means[i,:]         = self.x
            covariances[i,:,:] = self.P

        return (means, covariances)


    def rts_smoother(self, Xs, Ps, Qs=None, dt=None):
        assert len(Xs) == len(Ps)
        n, dim_x = Xs.shape

        if dt is None:
            dt = [self._dt] * n
        elif isscalar(dt):
            dt = [dt] * n

        if Qs is None:
            Qs = [self.Q] * n

        # smoother gain
        Ks = zeros((n,dim_x,dim_x))

        num_sigmas = self._num_sigmas

        xs, ps = Xs.copy(), Ps.copy()
        sigmas_f = zeros((num_sigmas, dim_x))

        for k in range(n-2,-1,-1):
            # create sigma points from state estimate, pass through state func
            sigmas = self.points_fn.sigma_points(xs[k], ps[k])
            for i in range(num_sigmas):
                sigmas_f[i] = self.fx(sigmas[i], dt[k])

            # compute backwards prior state and covariance
            xb = dot(self.Wm, sigmas_f)
            Pb = 0
            x = Xs[k]
            for i in range(num_sigmas):
                y = self.residual_x(sigmas_f[i], x)
                Pb += self.Wc[i] * outer(y, y)
            Pb += Qs[k]

            # compute cross variance
            Pxb = 0
            for i in range(num_sigmas):
                z = self.residual_x(sigmas[i], Xs[k])
                y = self.residual_x(sigmas_f[i], xb)
                Pxb += self.Wc[i] * outer(z, y)

            # compute gain
            K = dot(Pxb, inv(Pb))

            # update the smoothed estimates
            xs[k] += dot (K, self.residual_x(xs[k+1], xb))
            ps[k] += dot3(K, ps[k+1] - Pb, K.T)
            Ks[k] = K

        return (xs, ps, Ks)



if __name__ == "__main__":

    dt = 0.05

    radarUKF = UKF(dim_x=3, dim_z=1, dt=dt, kappa=0.)
    radarUKF.Q *= Q_discrete_white_noise(3, 1, .01)
    radarUKF.R *= 10
    radarUKF.x = np.array([0., 90., 1100.])
    radarUKF.P *= 100.

    t = np.arange(0, 20+dt, dt)
    n = len(t)
    xs = []
    rs = []
    for i in range(n):
        r = GetRadar(dt)
        rs.append(r)

        radarUKF.update(r, hx, fx)

        xs.append(radarUKF.x)

    xs = np.asarray(xs)

    plt.subplot(311)
    plt.plot(t, xs[:, 0])
    plt.title('distance')

    plt.subplot(312)
    plt.plot(t, xs[:, 1])
    plt.title('velocity')

    plt.subplot(313)
    plt.plot(t, xs[:, 2])
    plt.title('altitude')