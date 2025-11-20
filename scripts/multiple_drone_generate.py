import os

def generate_x2_drones(num_drones):
    body_template = """
    <body name="x2_{id}" pos="{x} {y} 0.3" childclass="x2">
      <freejoint/>
      <site name="imu_{id}" pos="0 0 .02"/>
      <geom material="phong3SG" mesh="X2_lowpoly" class="visual" quat="0 0 1 1"/>
      <geom class="collision" size=".06 .027 .02" pos=".04 0 .02"/>
      <geom class="collision" size=".06 .027 .02" pos=".04 0 .06"/>
      <geom class="collision" size=".05 .027 .02" pos="-.07 0 .065"/>
      <geom class="collision" size=".023 .017 .01" pos="-.137 .008 .065" quat="1 0 0 1"/>
      <geom name="rotor1_{id}" class="rotor" pos="-.14 -.18 .05" mass=".25"/>
      <geom name="rotor2_{id}" class="rotor" pos="-.14 .18 .05" mass=".25"/>
      <geom name="rotor3_{id}" class="rotor" pos=".14 .18 .08" mass=".25"/>
      <geom name="rotor4_{id}" class="rotor" pos=".14 -.18 .08" mass=".25"/>
      <geom size=".16 .04 .02" pos="0 0 0.02" type="ellipsoid" mass=".325" class="visual" material="invisible"/>
      <site name="thrust1_{id}" pos="-.14 -.18 .05"/>
      <site name="thrust2_{id}" pos="-.14 .18 .05"/>
      <site name="thrust3_{id}" pos=".14 .18 .08"/>
      <site name="thrust4_{id}" pos=".14 -.18 .08"/>
    </body>
    """

    actuator_template = """
    <motor class="x2" name="thrust1_{id}" site="thrust1_{id}" gear="0 0 1 0 0 -.0201"/>
    <motor class="x2" name="thrust2_{id}" site="thrust2_{id}" gear="0 0 1 0 0  .0201"/>
    <motor class="x2" name="thrust3_{id}" site="thrust3_{id}" gear="0 0 1 0 0 -.0201"/>
    <motor class="x2" name="thrust4_{id}" site="thrust4_{id}" gear="0 0 1 0 0  .0201"/>
    """

    sensor_template = """
    <gyro name="gyro_{id}" site="imu_{id}"/>
    <accelerometer name="acc_{id}" site="imu_{id}"/>
    <framepos name="pos_{id}" objtype="site" objname="imu_{id}"/>
    <framequat name="quat_{id}" objtype="site" objname="imu_{id}"/>
    <framelinvel name="vel_{id}" objtype="site" objname="imu_{id}"/>
    """

    worldbody_entries = []
    actuator_entries = []
    sensor_entries = []

    for i in range(num_drones):
        x = (i % 5) * 2.0
        y = (i // 5) * 2.0
        worldbody_entries.append(body_template.format(id=i, x=x, y=y))
        actuator_entries.append(actuator_template.format(id=i))
        sensor_entries.append(sensor_template.format(id=i))

    return "\n".join(worldbody_entries), "\n".join(actuator_entries), "\n".join(sensor_entries)

def save_multi_drone_xml(filename, num_drones):
    bodies_xml, actuators_xml, sensors_xml = generate_x2_drones(num_drones)

    base_xml = f"""<mujoco model="MultiDroneX2">
  <compiler autolimits="true" assetdir="assets"/>
  <option timestep="0.01" density="1.225" viscosity="1.8e-5"/>

  <default>
    <default class="x2">
        <geom mass="0"/>
        <motor ctrlrange="0 13"/>
        <mesh scale="0.01 0.01 0.01"/>
        <default class="visual">
          <geom group="2" type="mesh" contype="0" conaffinity="0"/>
        </default>
        <default class="collision">
          <geom group="3" type="box"/>
          <default class="rotor">
            <geom type="ellipsoid" size=".13 .13 .01"/>
          </default>
        </default>
        <site group="5"/>
    </default>
  </default>

  <asset>
    <texture type="2d" file="X2_lowpoly_texture_SpinningProps_1024.png"/>
    <material name="phong3SG" texture="X2_lowpoly_texture_SpinningProps_1024"/>
    <material name="invisible" rgba="0 0 0 0"/>
    <mesh class="x2" file="X2_lowpoly.obj"/>
  </asset>

  <worldbody>
    {bodies_xml}
  </worldbody>

  <actuator>
    {actuators_xml}
  </actuator>

  <sensor>
    {sensors_xml}
  </sensor>
</mujoco>"""
    

    if os.path.exists(filename):
        os.remove(filename)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(base_xml)


    print("File created successfully")

# Generate and save
if __name__ == "__main__":
  save_multi_drone_xml("mjcf/multiple_x22.xml", num_drones=5)
