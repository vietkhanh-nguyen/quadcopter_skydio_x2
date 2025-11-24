import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.multiple_drone_generate import save_multi_drone_xml
from mjc_sim.mjc_sim_1 import MujocoSim
from scenario.scenario_bearing import ScenarioBearingbasedConsensus
from scenario.scenario_drone_tracking import ScenarioDroneTracking
from utilities.build_map import build_map_summary

def mjc_sim_scenario_bearing_based():
    xml_path = '../mjcf/scene_multiple_x2.xml'
    simulation_time = 1000 #simulation time
    fps = 60
    num_drones = 12
    save_multi_drone_xml("mjcf/multiple_x2.xml", num_drones=num_drones)
    scenario = ScenarioBearingbasedConsensus()
    sim = MujocoSim(xml_path, num_drones, simulation_time, fps, scenario, None)
    sim.main_loop()

def mjc_sim_scenario_drone_tracking():
    xml_path = '../mjcf/scene.xml' #xml file (assumes this is in the same folder as this file)
    simulation_time = 100 #simulation time
    fps = 60
    num_drones = 1
    scenario = ScenarioDroneTracking()
    sim = MujocoSim(xml_path, num_drones, simulation_time, fps, scenario, None)
    sim.main_loop()


if __name__ == "__main__":
    # build_map_summary()
    mjc_sim_scenario_drone_tracking()