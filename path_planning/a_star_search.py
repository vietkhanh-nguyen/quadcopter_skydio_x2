import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from path_planning.a_star_obj import OpenList, ClosedList, AStarNode, Stack, NodeGrid3D
from path_planning.env import MapGridEnvironment3D
from utilities.build_map import generate_city_environment
from plots.my_plot import MyPlot

class AStartSearch3D:

    def __init__(self, map_env=None):
        """
        map_env: MapGridEnvironment object with 3D grid prepared
        """
        if map_env is None:
            raise ValueError("You must provide a MapGridEnvironment with a 3D occupancy grid")

        # Expect map_env.grid_map_3d to be a 3D numpy array (nz, ny, nx)
        self.grid = np.transpose(map_env.grid_map, (2, 1, 0))
        self.nz, self.grid_num_row, self.grid_num_col = self.grid.shape
        self.map = map_env
        self.start_node = None
        self.end_node = None
        self.path = []
        self.path_cons = []

    def define_start_end_node(self, start_pos_world, end_pos_world):
        """
        start_pos_world and end_pos_world are world coords (x, y, z).
        Converts to grid indices using map_env resolution.
        """
        start_node = NodeGrid3D(*self.map.cons_to_grid(start_pos_world))
        end_node = NodeGrid3D(*self.map.cons_to_grid(end_pos_world))
        self.start_node = start_node
        self.end_node = end_node

    def search(self):
        start_node = AStarNode(self.start_node, self.start_node, self.end_node, None)
        self.open_list = OpenList()
        self.closed_list = ClosedList()
        self.open_list.add_node(start_node)
        self.path = [(self.start_node.z, self.start_node.row, self.start_node.col)]

        maximum_iterations = 10000000
        for _ in range(maximum_iterations):
            if not self.open_list.open_list:
                return None

            current_node = self.open_list.pop_node()
            self.closed_list.add_node(current_node)

            if (current_node.node_pose.z == self.end_node.z and
                current_node.node_pose.row == self.end_node.row and
                current_node.node_pose.col == self.end_node.col):
                self.retrace_path(current_node)
                self.path_cons = self.map.grid_to_map_vector(self.path)
                
                return self.path_cons

            neighbours = self.get_neighbour(current_node)
            for node in neighbours:
                z, r, c = node.node_pose.z, node.node_pose.row, node.node_pose.col
                if (z < 0 or z >= self.nz or r < 0 or r >= self.grid_num_row or c < 0 or c >= self.grid_num_col):
                    continue
                if self.grid[z, r, c] != 0:
                    continue
                if self.closed_list.is_in_closed(node):
                    continue

                node.recalculate(current_node)
                self.open_list.is_not_in_open(node)

        return None

    def retrace_path(self, current_node):
        reversed_path = Stack()
        while current_node.parent_node is not None:
            reversed_path.push((current_node.node_pose.z,
                                current_node.node_pose.row,
                                current_node.node_pose.col))
            current_node = current_node.parent_node
        while not reversed_path.is_empty():
            self.path.append(reversed_path.pop())
        self.path = np.array(self.path)

    def get_neighbour(self, current_node):
        neighbours = []
        for dz in (-1, 0, 1):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dz == 0 and dr == 0 and dc == 0:
                        continue
                    z_new = current_node.node_pose.z + dz
                    r_new = current_node.node_pose.row + dr
                    c_new = current_node.node_pose.col + dc
                    neighbours.append(AStarNode(NodeGrid3D(z_new, r_new, c_new),
                                                self.start_node, self.end_node, current_node))
        return neighbours

    def inflate_obstacles(self, inflation_radius_m):
        radius_cells = int(np.ceil(inflation_radius_m / self.map.resolution))
        if radius_cells <= 0:
            return
        structure = np.ones((2 * radius_cells + 1,
                             2 * radius_cells + 1,
                             2 * radius_cells + 1))
        occupied = (self.grid != 0)
        inflated = binary_dilation(occupied, structure=structure).astype(int)
        self.grid = inflated
        self.map.grid_map_3d = inflated

def path_finding(start_pos, end_pos):
    city = np.load("path_planning/city_env.npy", allow_pickle=True).item()
    env = MapGridEnvironment3D(city["map_size"], city["resolution"], city["box_obs_list"])
    env.generate_voxel_map()
    a_star = AStartSearch3D(map_env=env)
    # start_pos = np.array([0.0, 0.0, 5.0])
    # end_pos = np.array([10.0, 10.0, 15.0])
    if env.is_inside_box(start_pos) or env.is_inside_box(end_pos):
        return False, None
    a_star.define_start_end_node(start_pos, end_pos)
    a_star.inflate_obstacles(3.0)
    path = a_star.search()
    if path is None:
        return False, None
    else:
        np.save("path_planning/astar_path.npy", path)
        return path, env

if __name__ == "__main__":

    city = np.load("path_planning/city_env.npy", allow_pickle=True).item()
    env = MapGridEnvironment3D(city["map_size"], city["resolution"], city["box_obs_list"])
    env.generate_voxel_map()
    a_star = AStartSearch3D(map_env=env)
    start_pos = np.array([0.0, 0.0, 5.0])
    end_pos = np.array([60.0, 60.0, 15.0])
    a_star.define_start_end_node(start_pos, end_pos)
    a_star.inflate_obstacles(3.0)
    path = a_star.search()
    print(path.shape)
    np.save("path_planning/astar_path.npy", path)
    myplot = MyPlot() 
    myplot.plot_3d_map(city["box_obs_list"], city["map_size"], elev=25, azim=45, waypoints=path)
    # print(path)
