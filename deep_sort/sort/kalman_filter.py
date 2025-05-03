import numpy as np
from filterpy.kalman import KalmanFilter

class CustomKalmanFilter:
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

    def predict(self):
        self.kf.predict()

    def update(self, bbox):
        z = self.convert_bbox_to_z(bbox)
        self.kf.update(z)

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        (x, y, s, r) formatından (x1, y1, x2, y2) formatına döner.
        """
        s = x[2]
        r = x[3]

        if s <= 0 or r <= 0:
            s = 1e-6
            r = 1.0

        w = np.sqrt(s * r)
        h = s / (w + 1e-6)

        x1 = x[0] - w / 2.
        y1 = x[1] - h / 2.
        x2 = x[0] + w / 2.
        y2 = x[1] + h / 2.

        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            x1, y1, x2, y2 = 0, 0, 0, 0

        return np.array([int(x1), int(y1), int(x2), int(y2)])
