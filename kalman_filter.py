import numpy as np

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        # Process noise covariance (Q)
        self.Q = process_variance

        # Measurement noise covariance (R)
        self.R = measurement_variance

        # Initial estimate covariance (P)
        self.P = 1.0

        # Initial estimate
        self.X = np.array([0.0, 0.0])

    def update(self, measurement):
        # Prediction step
        # For a constant velocity model, the state transition matrix is identity
        self.P = self.P + self.Q

        # Kalman Gain
        K = self.P / (self.P + self.R)

        # Update estimate
        self.X = self.X + K * (measurement - self.X)

        # Update covariance
        self.P = (1 - K) * self.P

        return self.X