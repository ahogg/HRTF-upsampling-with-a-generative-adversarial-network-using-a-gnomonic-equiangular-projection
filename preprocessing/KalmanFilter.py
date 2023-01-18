from numpy.linalg import inv
import numpy as np


class KalmanFilter:
    """Repurposed from https://github.com/ahogg/hogg2019-icassp-paper/blob/master/KalmanFilter.py by Aidan Hogg"""

    def __init__(self, x, P, H, Q, R):
        self.x = x  # estimated state
        self.P = P  # estimated covariance (measure of the accuracy of state estimate)
        self.H = H  # observation model
        self.Q = Q  # covariance of the process noise
        self.R = R  # covariance of the observation noise
        self.S = 0  # innovation covariance
        self.y = 0  # innovation
        self.k = np.array([[0], [0]])  # optimal kalman gain

    def prediction(self, F):
        # F is state transition model
        # 1. predict new state based on state transition model
        self.x = F.dot(self.x)
        # 2. predict new covariance based on state transition matrix + process noice
        self.P = F.dot(self.P).dot(F.T) + self.Q
        return

    def update(self, z):
        # z is observation
        # 1. update innovation (pre-fit residual) covariance
        self.S = self.H.dot(self.P).dot(self.H.T) + self.R
        # 2. update optimal kalman gain
        self.k = self.P.dot(self.H.T.dot(inv(self.S)))
        # 3. update innovation (pre-fit residual): difference between observed and predicted
        self.y = z - self.H.dot(self.x)
        # 4. update state estimate
        self.x = self.x + self.k.dot(self.y)
        # 5. update estimate covariance
        self.P = self.P - self.k.dot(self.H.dot(self.P))
        return

    def get_err_variance(self):
        return self.P.item(0)

    def get_err_covariance(self):
        return self.P

    def get_inno_covariance(self):
        # innovation covariance
        return self.S

    def set_err_covariance(self, P):
        self.P = P
        return

    def get_kalman_gain(self):
        return self.k

    def get_state(self):
        return self.x.item(0)

    def get_post_fit_residual(self):
        return self.k.dot(self.y).item(0)
