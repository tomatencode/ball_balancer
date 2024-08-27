import numpy as np

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        # Process noise variance (scalar value for acceleration noise)
        self.q = process_variance

        # Measurement noise covariance (R)
        self.R = measurement_variance * np.eye(4)  # Now 4x4 to handle both position and velocity measurements

        # Initial estimate covariance (P)
        self.P = np.eye(6)

        # Initial state estimate [x, y, vx, vy, ax, ay]
        self.X = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Measurement matrix (H)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # x position
            [0, 1, 0, 0, 0, 0],  # y position
            [0, 0, 1, 0, 0, 0],  # x velocity
            [0, 0, 0, 1, 0, 0]   # y velocity
        ])

    def compute_F_and_Q(self, dt):
        """Compute the state transition matrix (F) and process noise covariance matrix (Q) based on the current dt."""
        F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        Q = self.q * np.array([
            [0.25*dt**4, 0, 0.5*dt**3, 0, 0.5*dt**2, 0],
            [0, 0.25*dt**4, 0, 0.5*dt**3, 0, 0.5*dt**2],
            [0.5*dt**3, 0, dt**2, 0, dt, 0],
            [0, 0.5*dt**3, 0, dt**2, 0, dt],
            [0.5*dt**2, 0, dt, 0, 1, 0],
            [0, 0.5*dt**2, 0, dt, 0, 1]
        ])

        return F, Q

    def update(self, measurement, acceleration, dt):
        # Compute the state transition matrix F and process noise covariance Q based on the current dt
        F, Q = self.compute_F_and_Q(dt)

        # Update the state transition matrix with known acceleration
        self.X[4:] = acceleration

        # Prediction step
        self.X = np.dot(F, self.X)
        self.P = np.dot(F, np.dot(self.P, F.T)) + Q

        # Kalman Gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update estimate with the new measurement (position and velocity)
        y = measurement - np.dot(self.H, self.X)
        self.X = self.X + np.dot(K, y)

        # Update covariance matrix
        I = np.eye(6)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        # Return the estimated position and velocity
        return self.X[:2], self.X[2:4]  # [x, y] [vx, vy]