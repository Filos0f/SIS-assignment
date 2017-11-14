from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import scipy.linalg as linalg
from numpy import dot, zeros, eye
from filterpy.common import setter, setter_1d, setter_scalar, dot3


class ExtendedKalmanFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self._x = zeros((dim_x,1)) # state
        self._P = eye(dim_x)       # uncertainty covariance
        self._B = 0                # control transition matrix
        self._F = 0                # state transition matrix
        self._R = eye(dim_z)       # state uncertainty
        self._Q = eye(dim_x)       # process uncertainty
        self._y = zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)


    def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):
        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        F = self._F
        B = self._B
        P = self._P
        Q = self._Q
        R = self._R
        x = self._x

        H = HJacobian(x, *args)

        # predict step
        x = dot(F, x) + dot(B, u)
        P = dot3(F, P, F.T) + Q

        # update step
        S = dot3(H, P, H.T) + R
        K = dot3(P, H.T, linalg.inv (S))
        self._K = K

        self._x = x + dot(K, (z - Hx(x, *hx_args)))

        I_KH = self._I - dot(K, H)
        self._P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    def update(self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
               residual=np.subtract):
        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        P = self._P
        if R is None:
            R = self._R
        elif np.isscalar(R):
            R = eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        x = self._x

        H = HJacobian(x, *args)

        S = dot3(H, P, H.T) + R
        K = dot3(P, H.T, linalg.inv (S))
        self._K = K

        hx =  Hx(x, *hx_args)
        y = residual(z, hx)
        self._y = y
        self._x = x + dot(K, y)

        I_KH = self._I - dot(K, H)
        self._P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)


    def predict_x(self, u=0):
        self._x = dot(self._F, self._x) + dot(self._B, u)


    def predict(self, u=0):
        self.predict_x(u)
        self._P = dot3(self._F, self._P, self._F.T) + self._Q


    @property
    def Q(self):
        """ Process uncertainty matrix"""
        return self._Q


    @Q.setter
    def Q(self, value):
        """ Process uncertainty matrix"""
        self._Q = setter_scalar(value, self.dim_x)


    @property
    def P(self):
        """ state covariance matrix"""
        return self._P


    @P.setter
    def P(self, value):
        """ state covariance matrix"""
        self._P = setter_scalar(value, self.dim_x)


    @property
    def R(self):
        """ measurement uncertainty"""
        return self._R


    @R.setter
    def R(self, value):
        """ measurement uncertainty"""
        self._R = setter_scalar(value, self.dim_z)


    @property
    def F(self):
        """State Transition matrix"""
        return self._F


    @F.setter
    def F(self, value):
        """State Transition matrix"""
        self._F = setter(value, self.dim_x, self.dim_x)


    @property
    def B(self):
        """ control transition matrix"""
        return self._B


    @B.setter
    def B(self, value):
        """ control transition matrix"""
        self._B = setter(value, self.dim_x, self.dim_u)


    @property
    def x(self):
        """ state estimate vector """
        return self._x

    @x.setter
    def x(self, value):
        """ state estimate vector """
        self._x = setter_1d(value, self.dim_x)

    @property
    def K(self):
        """ Kalman gain """
        return self._K

    @property
    def y(self):
        """ measurement residual (innovation) """
        return self._y

    @property
    def S(self):
        """ system uncertainty in measurement space """
        return self._S


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    wheelDiam = 55/10
    from numpy import genfromtxt

    my_dataEncoder = genfromtxt('EncoderData.csv', delimiter='')
    # Converting to distance in meters from encoder degrees data
    observed_Encoder = my_dataEncoder*(3.14/180)*wheelDiam
    np.savetxt('EncoderData_new.csv',observed_Encoder,delimiter=',')
    my_dataC = genfromtxt('USData.csv', delimiter='')
    # After careful observation of the data it was split into three regions 
    # with high sidrupts in sonar distance measurements
    observed_Sonar1 = my_dataC[0:128]-min(my_dataC[0:128])
    observed_Sonar1 = observed_Sonar1[::-1]
    observed_Sonar2 = my_dataC[128:213]-min(my_dataC[128:213])
    observed_Sonar2 = observed_Sonar2[::-1]
    observed_Sonar3 = my_dataC[213:340]-min(my_dataC[213:340])
    observed_Sonar3 = observed_Sonar3[::-1]

    observed_Sonar2 = observed_Sonar2 + max(observed_Sonar1)
    observed_Sonar3 = observed_Sonar3 + max(observed_Sonar2)
    observed_Sonar = np.append(observed_Sonar1,observed_Sonar2)
    observed_Sonar = np.append(observed_Sonar,observed_Sonar3)
    
    observed_Res = np.insert(observed_Sonar, np.arange(len(observed_Encoder)), observed_Encoder)
    np.savetxt('USData_new.csv',observed_Encoder,delimiter=',')
    
    time = np.arange(0,observed_Res.shape[0],1)
    timeS = np.arange(0,observed_Res.shape[0],0.5)

    iteration_count = observed_Res.shape[0]
          
    measurement_standard_deviation = np.std([observed_Res])
    process_variance = 0.55
    estimated_measurement_variance = measurement_standard_deviation ** 2
    
    sensorNoise = np.array([[math.pow(3,2),0],[0,math.pow(2,2)]])
    stateTransfer = lambda x: np.array([[math.pow(x[0][0],1.01)],[math.pow(x[1][0],.99)+5]]) 
    sensorTransfer = lambda x: x 
    processNoise = np.array([[0],[0]])
    processNoiseCovariance = np.array([[.1,0],[0,.1]])

    initialReading = np.array([[observed_Sonar[1]],[observed_Encoder[1]]])
    kf = ExtendedKalman(initialReading,2,sensorNoise,stateTransfer,sensorTransfer, processNoise, processNoiseCovariance)
    #kalman_filter = Kalman_Filter(process_variance, estimated_measurement_variance)
    kalman_out = []

    for iteration in range(2, 340):
            #kalman_filter.input_observed_measurement(observed_Res[iteration])
            #kalman_out.append(kalman_filter.get_estimated_measurement())
            
            reading = np.array([observed_Sonar[iteration],observed_Encoder[iteration]])
            kf.predict()
            kf.update(reading)

            #grab result for this iteration and figure out a resistance value
            myEstimate = kf.getEstimate()
            voltage_guess = myEstimate[0][0]
            current_guess = myEstimate[1][0]

    import pylab
    pylab.figure()
    pylab.plot(observed_Res, 'g,', label='Upper - Encoder, lower - Sonar')
    pylab.plot(kalman_out, 'r-', label='Kalman')
    pylab.legend()
    pylab.xlabel('time step')
    pylab.ylabel('Distance, (cm)')
    pylab.savefig('KF.png',dpi=300)
    pylab.show()