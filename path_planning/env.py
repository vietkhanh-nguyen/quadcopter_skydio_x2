import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from plots.my_plot import MyPlot

class MapGridEnvironment3D:

    def __init__(self, map_size, resolution, box_obs_list):
        """
        map_size = (x_min, x_max, y_min, y_max, z_min, z_max)
        resolution = voxel size (meters)
        box_obs_list = list of 3D boxes [xmin, xmax, ymin, ymax, zmin, zmax]
        """
        self.map_size = map_size
        self.resolution = resolution

        self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max = map_size

        self.box_obs_list = np.array(box_obs_list)
    # -----------------------------------------
    # GRID CONVERSION
    # -----------------------------------------
    def cons_to_grid(self, xyz):
        """
        Convert world coordinates [x, y, z] to grid indices [z, row, col] for A*.
        """
        x, y, z = xyz
        col = int(np.floor((x - self.x_min) / self.resolution))
        row = int(np.floor((y - self.y_min) / self.resolution))
        z_idx = int(np.floor((z - self.z_min) / self.resolution))
        return np.array([z_idx, row, col], dtype=int)

    def grid_to_cons(self, ijk):
        """
        Convert grid indices [z, row, col] to world coordinates [x, y, z].
        """
        z_idx, row, col = ijk
        x = col * self.resolution + self.x_min
        y = row * self.resolution + self.y_min
        z = z_idx * self.resolution + self.z_min
        return np.array([x, y, z], dtype=float)

    def grid_to_map_vector(self, grid_points):
        """
        Convert a list of grid indices [(z, row, col), ...] to world coordinates.
        """
        world_points = [self.grid_to_cons(p) for p in grid_points]
        return np.array(world_points)


    # -----------------------------------------
    # 3D COLLISION FUNCTIONS
    # -----------------------------------------
    def is_inside_box(self, point):
        x, y, z = point
        for box in self.box_obs_list:
            xmin, xmax, ymin, ymax, zmin, zmax = box
            if (
                xmin <= x <= xmax and
                ymin <= y <= ymax and
                zmin <= z <= zmax
            ):
                return True
        return False
    
    # compute closest point on a 3D box to a point in space
    def closest_point_on_box(self, point, box):
        px, py, pz = point
        xmin, xmax, ymin, ymax, zmin, zmax = box

        cx = np.clip(px, xmin, xmax)
        cy = np.clip(py, ymin, ymax)
        cz = np.clip(pz, zmin, zmax)

        return np.array([cx, cy, cz])
    
    def closest_point_on_box_batch(self, points, box):
        """
        Vectorized closest-point computation.
        points: (n_agents, 3)
        box: [xmin, xmax, ymin, ymax, zmin, zmax]
        Returns: (n_agents, 3)
        """
        xmin, xmax, ymin, ymax, zmin, zmax = box

        closest = np.empty_like(points)
        closest[:, 0] = np.clip(points[:, 0], xmin, xmax)
        closest[:, 1] = np.clip(points[:, 1], ymin, ymax)
        closest[:, 2] = np.clip(points[:, 2], zmin, zmax)

        return closest

    # --------------------------------------------------------
    # 3D REPULSIVE FIELD
    # --------------------------------------------------------
    def compute_repulsive_velocity_multi(self, agents_pos,
                                     influence_distance=3,
                                     eta=1.0):
        """
        Vectorized repulsive velocity computation for multiple agents.
        
        agents_pos: (n_agents, 3)
        Returns v_rep:   (n_agents, 3)
        """
        n_agents = agents_pos.shape[0]
        v_rep = np.zeros((n_agents, 3))

        for box in self.box_obs_list:
            # get closest point for all agents (vectorized call)
            closest = self.closest_point_on_box_batch(agents_pos, box)   # (n_agents, 3)

            diff = agents_pos - closest
            d = np.linalg.norm(diff, axis=1)  # (n_agents,)

            # --- Agents inside obstacle -----------------------------------
            inside = d < 1e-6
            if np.any(inside):
                px, py, pz = agents_pos[inside].T
                xmin, xmax, ymin, ymax, zmin, zmax = box

                dx_min = px - xmin
                dx_max = xmax - px
                dy_min = py - ymin
                dy_max = ymax - py
                dz_min = pz - zmin
                dz_max = zmax - pz

                distances = np.vstack([dx_min, -dx_max, dy_min, -dy_max, dz_min, -dz_max]).T
                axis = np.argmax(np.abs(distances), axis=1)

                push = np.zeros((inside.sum(), 3))
                for k in range(axis.size):
                    if axis[k] == 0: push[k,0] =  1
                    if axis[k] == 1: push[k,0] = -1
                    if axis[k] == 2: push[k,1] =  1
                    if axis[k] == 3: push[k,1] = -1
                    if axis[k] == 4: push[k,2] =  1
                    if axis[k] == 5: push[k,2] = -1

                v_rep[inside] += eta * 10 * push

            # --- Standard repulsive field ---------------------------------
            mask = (d < influence_distance) & (~inside)
            if np.any(mask):
                d_mask = d[mask]
                diff_mask = diff[mask]

                v_rep[mask] += eta * (1/d_mask - 1/influence_distance)[:,None] * \
                            (1/d_mask**2)[:,None] * \
                            (diff_mask / d_mask[:,None])

        return v_rep


    # -----------------------------------------
    # BUILD VOXEL MAP (3D OCCUPANCY GRID)
    # -----------------------------------------
    def generate_voxel_map(self):
        self.nx = int((self.x_max - self.x_min) / self.resolution)
        self.ny = int((self.y_max - self.y_min) / self.resolution)
        self.nz = int((self.z_max - self.z_min) / self.resolution)

        grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)

        # -----------------------------
        # HANDLE BOX OBSTACLES
        # -----------------------------
        for box in self.box_obs_list:
            xmin, xmax, ymin, ymax, zmin, zmax = box

            gx_min = int(np.floor((xmin - self.x_min) / self.resolution))
            gx_max = int(np.ceil( (xmax - self.x_min) / self.resolution))

            gy_min = int(np.floor((ymin - self.y_min) / self.resolution))
            gy_max = int(np.ceil( (ymax - self.y_min) / self.resolution))

            gz_min = int(np.floor((zmin - self.z_min) / self.resolution))
            gz_max = int(np.ceil( (zmax - self.z_min) / self.resolution))

            grid[gx_min:gx_max+1, gy_min:gy_max+1, gz_min:gz_max+1] = 1

        self.grid_map = grid
        return grid

    # -----------------------------------------
    # EXPORT
    # -----------------------------------------
    def save_numpy(self, filename="map_3d.npy"):
        if not hasattr(self, "grid_map"):
            raise ValueError("Call generate_voxel_map() first.")
        np.save(filename, self.grid_map)
        print(f"Saved voxel map → {filename}")








if __name__ == "__main__":
    city = np.load("path_planning/city_env.npy", allow_pickle=True).item()
    env = MapGridEnvironment3D(city["map_size"], city["resolution"], city["box_obs_list"])
    myplot = MyPlot() 
    grid = env.generate_voxel_map()
    env.save_numpy("voxel_map.npy")
    myplot.plot_3d_map(city["box_obs_list"], city["map_size"])
