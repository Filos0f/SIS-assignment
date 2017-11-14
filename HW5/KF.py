from __future__ import (absolute_import, division, unicode_literals)
from filterpy.common import setter, setter_1d, setter_scalar, dot3
from filterpy.stats import logpdf
import math
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import scipy.linalg as linalg
import sys


class KalmanFilter(object):
    def __init__(self, dim_x, dim_z, dim_u=0):

        assert dim_x > 0
        assert dim_z > 0
        assert dim_u >= 0

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x,1)) # state
        self.P = eye(dim_x)       # uncertainty covariance
        self.Q = eye(dim_x)       # process uncertainty
        self.B = 0                # control transition matrix
        self.F = 0                # state transition matrix
        self.H = 0                 # Measurement function
        self.R = eye(dim_z)        # state uncertainty
        self._alpha_sq = 1.        # fading memory control
        self.M = 0                 # process-measurement cross correlation

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros(self.x.shape) # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty

        # identity matrix. Do not alter this.
        self.I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_pred = zeros((dim_x,1))
        self.P_pred = eye(dim_x)


    def update(self, z, R=None, H=None):

        if z is None:
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        if H is None:
            H = self.H
        P = self.P
        x = self.x

        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if x.ndim==1 and shape(z) == (1,1):
            z = z[0]

        if shape(z) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

        # y = z - Hx
        # error (residual) between measurement and prediction
        Hx = dot(H, x)

        assert shape(Hx) == shape(z) or (shape(Hx) == (1,1) and shape(z) == (1,)), \
               'shape of z should be {}, but it is {}'.format(
               shape(Hx), shape(z))
        self.y = z - Hx

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot3(H, P, H.T) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot3(P, H.T, linalg.inv(self.S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self.I - dot(self.K, H)
        self.P = dot3(I_KH, P, I_KH.T) + dot3(self.K, R, self.K.T)

        self.log_likelihood = logpdf(z, dot(H, x), self.S)


    def update_correlated(self, z, R=None, H=None):

        if z is None:
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        if H is None:
            H = self.H
        x = self.x
        P = self.P
        M = self.M

        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if x.ndim==1 and shape(z) == (1,1):
            z = z[0]

        if shape(z) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, x)

        # project system uncertainty into measurement space
        self.S = dot3(H, P, H.T) + dot(H, M) + dot(M.T, H.T) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(dot(P, H.T) + M, linalg.inv(self.S))

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = x + dot(self.K, self.y)
        self.P = P - dot(self.K, dot(H, P) + M.T)

        # compute log likelihood
        self.log_likelihood = logpdf(z, dot(H, x), self.S)


    def predict(self, u=0, B=None, F=None, Q=None):

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        self.x = dot(F, self.x) + dot(B, u)

        # P = FPF' + Q
        self.P = self._alpha_sq * dot3(F, self.P, F.T) + Q

        self.x_pred = self.x[:]
        self.P_pred = self.P[:]
