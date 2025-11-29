def write_waypoints_xml(path, filename="mjcf/waypoints.xml", size=0.05, color=(1,0,0,1)):
    with open(filename, "w") as f:
        # start the MJCF file
        f.write('<mujoco>\n')
        f.write('  <worldbody>\n')
        
        if path is None or len(path) == 0:
            # add a dummy invisible geom
            f.write('    <geom name="dummy" type="sphere" size="0.001" rgba="0 0 0 0" pos="0 0 -10"/>\n')
            print(f"Waypoints XML '{filename}' cleared with dummy geom.")
        else:
            for i, (x, y, z) in enumerate(path):
                rgba_str = " ".join(map(str, color))
                # contype=0 and conaffinity=0 makes it non-contactable
                f.write(f'    <geom name="wp_{i}" type="sphere" size="{size}" rgba="{rgba_str}" pos="{x} {y} {z}" contype="0" conaffinity="0"/>\n')
            print(f"Waypoints XML '{filename}' written with {len(path)} points (non-contactable).")
        
        # close the worldbody and mujoco tags
        f.write('  </worldbody>\n')
        f.write('</mujoco>\n')
