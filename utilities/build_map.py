import numpy as np

def generate_city_environment(
    num_blocks_x=4,
    num_blocks_y=4,
    block_size=20,
    road_width=10,
    min_building_size=4,
    max_building_size=12,
    min_height=5,
    max_height=25,
    seed=None,
    clear_spawn_area=(0, 20, 0, 20)  # (xmin, xmax, ymin, ymax)
):
    """
    Generate a city-like environment as a list of rectangular 3D obstacles.
    Keeps the spawn area clear for UAV.
    """
    if seed is not None:
        np.random.seed(seed)

    obstacles = []
    lot_size = block_size - road_width
    spawn_xmin, spawn_xmax, spawn_ymin, spawn_ymax = clear_spawn_area

    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            block_x0 = i * block_size + road_width / 2
            block_y0 = j * block_size + road_width / 2

            num_buildings = np.random.randint(5, 10)

            for _ in range(num_buildings):
                bw = np.random.uniform(min_building_size, max_building_size)
                bd = np.random.uniform(min_building_size, max_building_size)

                # Random placement inside block
                x0 = block_x0 + np.random.uniform(0, lot_size - bw)
                y0 = block_y0 + np.random.uniform(0, lot_size - bd)
                x1 = x0 + bw
                y1 = y0 + bd

                # Check if building intersects spawn area
                if (x1 > spawn_xmin and x0 < spawn_xmax) and (y1 > spawn_ymin and y0 < spawn_ymax):
                    continue  # skip this building

                h = np.random.uniform(min_height, max_height)
                obstacles.append((x0, x1, y0, y1, 0, h))

    return obstacles

def export_obstacles_to_xml(box_obs_list, filename="mjcf/obstacles.xml", geom_type="box", material="obstacle_material"):

    lines = ['<mujoco>\n', '  <worldbody>\n']

    for i, box in enumerate(box_obs_list):
        xmin, xmax, ymin, ymax, zmin, zmax = box
        # MuJoCo <geom> center = midpoint, size = half-extents
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        z_center = (zmin + zmax) / 2
        x_size = (xmax - xmin) / 2
        y_size = (ymax - ymin) / 2
        z_size = (zmax - zmin) / 2

        lines.append(
            f'    <geom name="obstacle_{i}" type="{geom_type}" pos="{x_center} {y_center} {z_center}" '
            f'size="{x_size} {y_size} {z_size}" material="{material}"/>\n'
        )

    lines.append('  </worldbody>\n')
    lines.append('</mujoco>\n')

    with open(filename, "w") as f:
        f.writelines(lines)

    print(f"Exported {len(box_obs_list)} obstacles → {filename}")

def build_map_summary():

    map_size = (0, 80, 0, 80, 0, 20)
    resolution = 1

    box_obs_list = generate_city_environment(
        num_blocks_x=4,
        num_blocks_y=4,
        seed=42
    )

    city = {
        "map_size": map_size,
        "resolution": resolution,
        "box_obs_list": box_obs_list
    }
    export_obstacles_to_xml(box_obs_list)
    np.save("path_planning/city_env.npy", city)


if __name__ == "__main__":

    build_map_summary()

    