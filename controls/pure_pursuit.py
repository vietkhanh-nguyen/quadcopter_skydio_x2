import numpy as np

class PurePursuit:

    def __init__(self, look_ahead_dist, waypoints, alpha):
        self.look_ahead_dist = look_ahead_dist
        self.waypoints = waypoints
        self.ref_filtered = None         # filtered reference point
        self.alpha = alpha     
        self.waypoints_len = len(waypoints)
        self.goal_flag = False   

    def look_ahead_point(self, pos_cur):
        # 1. Compute the raw lookahead point
        dists = np.linalg.norm(self.waypoints - pos_cur[:3], axis=1)
        closest_idx = np.argmin(dists)

        cum_dist = 0.0
        for i in range(closest_idx, len(self.waypoints) - 1):
            segment = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
            cum_dist += segment
            if cum_dist >= self.look_ahead_dist:
                raw_ref = self.waypoints[i+1]
                break
        else:
            raw_ref = self.waypoints[-1]

        # 2. Apply low-pass filter
        if self.ref_filtered is None:
            self.ref_filtered = raw_ref.copy()
        else:
            self.ref_filtered = (
                self.alpha * self.ref_filtered + (1 - self.alpha) * raw_ref
            )

        goal_tol = 0.1  # meters
        if np.linalg.norm(pos_cur[:3] - self.waypoints[-1]) < goal_tol:
            self.goal_flag = True

        return self.ref_filtered
