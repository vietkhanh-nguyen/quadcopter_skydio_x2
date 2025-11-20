import numpy as np

class PurePursuit:

    def __init__(self, look_ahead_dist, waypoints):
         self.look_ahead_dist = look_ahead_dist
         self.waypoints = waypoints

    def look_ahead_point(self, pos_cur):
            dists = np.linalg.norm(self.waypoints - pos_cur[:3], axis=1)
            closest_idx = np.argmin(dists)
            cum_dist = 0.0
            for i in range(closest_idx, len(self.waypoints )-1):
                segment = np.linalg.norm(self.waypoints [i+1] - self.waypoints [i])
                cum_dist += segment
                if cum_dist >= self.look_ahead_dist :
                    return self.waypoints [i+1]

            # If look-ahead distance goes beyond path, return last point
            return self.waypoints [-1]