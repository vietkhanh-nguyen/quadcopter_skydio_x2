import numpy as np
import matplotlib.pyplot as plt
from plots.my_plot import plot_3d_map

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

    grid = env.generate_voxel_map()
    env.save_numpy("voxel_map.npy")
    plot_3d_map(city["box_obs_list"], city["map_size"])
