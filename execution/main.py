import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.multiple_drone_generate import save_multi_drone_xml
from mjc_simulate.mjc_sim import MujocoSim
from mjc_simulate.mjc_render import MujocoRender
from scenario.scenario_bearing import ScenarioBearingbasedConsensus
from scenario.scenario_drone_tracking import ScenarioDroneTracking
from utilities.build_map import build_map_summary

def mjc_render_scenario_bearing_based():
    xml_path = '../mjcf/scene_multiple_x2.xml'
    simulation_time = 40 #simulation time
    time_step = 0.01
    fps = 60
    num_drones = 12
    # build_map_summary()
    save_multi_drone_xml("mjcf/multiple_x2.xml", num_drones=num_drones)
    scenario = ScenarioBearingbasedConsensus()
    render = MujocoRender(xml_path, num_drones, simulation_time, time_step, fps, scenario)
    render.main_loop()

def mjc_sim_scenario_bearing_based():
    xml_path = '../mjcf/scene_multiple_x2.xml'
    simulation_time = 1000 #simulation time
    time_step = None
    fps = 60
    num_drones = 12
    plot = None
    # build_map_summary()
    save_multi_drone_xml("mjcf/multiple_x2.xml", num_drones=num_drones)
    scenario = ScenarioBearingbasedConsensus()
    sim = MujocoSim(xml_path, num_drones, simulation_time, time_step, fps, scenario, plot)
    sim.main_loop()

def mjc_sim_scenario_drone_tracking():
    xml_path = '../mjcf/scene.xml' #xml file (assumes this is in the same folder as this file)
    simulation_time = 100 #simulation time
    time_step = None
    fps = 60
    num_drones = 1
    # build_map_summary()
    plot = None
    scenario = ScenarioDroneTracking()
    sim = MujocoSim(xml_path, num_drones, simulation_time, time_step, fps, scenario, plot)
    sim.main_loop()


if __name__ == "__main__":
    mjc_render_scenario_bearing_based()