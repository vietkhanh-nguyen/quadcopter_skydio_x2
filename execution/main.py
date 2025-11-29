import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.multiple_drone_generate import save_multi_drone_xml
from mjc_simulate.mjc_sim import MujocoSim
from mjc_simulate.mjc_render import MujocoRender
from scenario.scenario_bearing import ScenarioBearingbasedConsensus
from scenario.scenario_drone_tracking import ScenarioDroneTracking
from scenario.scenario_bearing_tracking import ScenarioBearingbasedTrackingConsensus
from scenario.scenario_bearing_center_tracking import ScenarioBearingbasedCenterTrackingConsensus
from utilities.build_map import build_map_summary


def mjc_sim_scenario_bearing_center_tracking_based():
    xml_path = '../mjcf/scene_multiple_x2.xml'
    simulation_time = 155 #simulation time
    time_step = None
    fps = 60
    num_drones = 12
    plot = None
    render_video = True
    # build_map_summary()
    save_multi_drone_xml("mjcf/multiple_x2.xml", num_drones=num_drones)
    scenario = ScenarioBearingbasedCenterTrackingConsensus()
    if render_video:
        render = MujocoRender(xml_path, num_drones, simulation_time, time_step, fps, scenario)
        render.main_loop()
    else:
        sim = MujocoSim(xml_path, num_drones, simulation_time, time_step, fps, scenario, plot)
        sim.main_loop()

def mjc_sim_scenario_bearing_tracking_based():
    xml_path = '../mjcf/scene_multiple_x2.xml'
    simulation_time = 150 #simulation time
    time_step = None
    fps = 25
    num_drones = 12
    plot = None
    render_video = True
    # build_map_summary()
    save_multi_drone_xml("mjcf/multiple_x2.xml", num_drones=num_drones)
    scenario = ScenarioBearingbasedTrackingConsensus()
    if render_video:
        render = MujocoRender(xml_path, num_drones, simulation_time, time_step, fps, scenario)
        render.main_loop()
    else:
        sim = MujocoSim(xml_path, num_drones, simulation_time, time_step, fps, scenario, plot)
        sim.main_loop()


def mjc_sim_scenario_bearing_based():
    xml_path = '../mjcf/scene_multiple_x2.xml'
    simulation_time = 78 #simulation time
    time_step = None
    fps = 60
    num_drones = 12
    plot = None
    render_video = False
    # build_map_summary()
    save_multi_drone_xml("mjcf/multiple_x2.xml", num_drones=num_drones)
    scenario = ScenarioBearingbasedConsensus()
    if render_video:
        render = MujocoRender(xml_path, num_drones, simulation_time, time_step, fps, scenario)
        render.main_loop()
    else:
        sim = MujocoSim(xml_path, num_drones, simulation_time, time_step, fps, scenario, plot)
        sim.main_loop()

def mjc_sim_scenario_drone_tracking():
    xml_path = '../mjcf/scene.xml' #xml file (assumes this is in the same folder as this file)
    simulation_time = 200 #simulation time
    time_step = None
    fps = 25
    num_drones = 1
    render_video = False
    plot = None
    # build_map_summary()
    scenario = ScenarioDroneTracking()
    if render_video:
        render = MujocoRender(xml_path, num_drones, simulation_time, time_step, fps, scenario)
        render.main_loop()
    else:
        sim = MujocoSim(xml_path, num_drones, simulation_time, time_step, fps, scenario, plot)
        sim.main_loop()


if __name__ == "__main__":
    mjc_sim_scenario_bearing_center_tracking_based()