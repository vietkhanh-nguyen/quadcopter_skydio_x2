import numpy as np
import mujoco as mj
from controls.quadcopter_controller import QuadcopterPIDController
from controls.consensus_controller import MultiAgentConsensus
from controls.pure_pursuit import PurePursuit
from path_planning.env import MapGridEnvironment3D
from path_planning.a_star_search import path_finding
from utilities.gen_multi_agent_graph import *
from utilities.visualize_waypoint import write_waypoints_xml

class ScenarioBearingbasedTrackingConsensus:

    def __init__(self):
        self.name = "Drones Formation using Bearing-based Consensus Algorithm"
        start_pos = np.array([0.0, 0.0, 5.0])
        end_pos = np.array([40.0, 40.0, 15.0])
        self.path, self.env = path_finding(start_pos, end_pos)
        self.e_bearing_tol = 1
        self.formation_scale = 0.35
        write_waypoints_xml(self.path, "mjcf/waypoints.xml")

    def init(self, sim, model, data):
        write_waypoints_xml(None, "mjcf/waypoints.xml")
        sim.cam.azimuth = 60
        sim.cam.elevation = -38
        sim.cam.distance =  9.5
        sim.cam.lookat =np.array([0.0 , 0.0 , 0.0])
        

        if self.path is None:
            self.path = sim.pos_ref

        self.path_tracking = PurePursuit(look_ahead_dist=2, waypoints=self.path, alpha=0.95)
        self.controllers = []
        self.formation_controller = MultiAgentConsensus(sim.num_drones, K=2, graph_type="complete")

        for _ in range(sim.num_drones):
            self.controllers.append(QuadcopterPIDController(sim.time_step))

        self.leader_controller = []
        for _ in range(2):
            self.leader_controller.append(QuadcopterPIDController(sim.time_step))

        self.tracking_flag = False
        self.formation_flag = False
        self.not_landing_flag = True
        
        self.t_prev = sim.data.time
        self.v_ref = np.zeros((sim.num_drones, 3))
        
        self.X_virtual = np.array([-1, -1, 0])
        self.dX_virtual = np.array([0, 0, 0])
        self.ddX_virtual = np.array([0, 0, 0])

        self.operation_mode = "take_off" 

    def update(self, sim, model, data):

        n_drones = sim.num_drones
        dt = sim.data.time - self.t_prev
        # --- Read current states ---
        pos_full = np.zeros((n_drones, 3))
        quat_full = np.zeros((n_drones, 4))
        linvel_full = np.zeros((n_drones, 3))
        angvel_full = np.zeros((n_drones, 3))
        state_full = np.zeros((n_drones, 13))
        acc_full = np.zeros((n_drones, 3))        
        
        for i in range(n_drones):
            
            pos_full[i, :] = np.array(data.sensor(f'pos_{i}').data)
            quat_full[i, :] = np.array(data.sensor(f'quat_{i}').data)
            linvel_full[i, :] = np.array(data.sensor(f'vel_{i}').data)
            angvel_full[i, :] = np.array(data.sensor(f'gyro_{i}').data)

            vel = np.hstack((linvel_full[i, :], angvel_full[i, :]))
            state_full[i, :] = np.concatenate([pos_full[i, :], quat_full[i, :], vel])

            body_acc = np.array(data.sensor(f'acc_{i}').data) 
            acc_full[i, :] = self.controllers[i].linear_acc_world(quat_full[i, :], body_acc)
            
                
        e_bearing = self.formation_controller.compute_bearing_error(pos_full)
        e_bearing_norm = np.linalg.norm(e_bearing)
        v_rep = self.env.compute_repulsive_velocity_multi(pos_full, influence_distance=1, eta=0.001)


        match self.operation_mode:

            case "take_off":
                self.v_ref = np.zeros_like(pos_full)
                self.altitude_ref = 5 * np.ones(sim.num_drones)
                
                if np.all(np.abs(pos_full[:, 2] - self.altitude_ref) < 0.05):
                    self.operation_mode = "formation_icosahedron"
                    self.altitude_ref = pos_full[:, 2]
                    

            case "formation_icosahedron":
                control_consensus = self.formation_controller.consensus_law(pos_full)
                self.v_ref = control_consensus + v_rep       
                self.altitude_ref += self.v_ref[:, 2] * dt

                if e_bearing_norm < self.e_bearing_tol:
                    self.operation_mode = "cruise"
                    self.altitude_ref = pos_full[:, 2]
                    self.formation_scale = 0.3
                
            case "cruise":
                control_consensus = self.formation_controller.consensus_leader_vel_varying_law(
                    pos_full, linvel_full, acc_full, kp=100, kv=200, ka=2, kadv=20
                )
                self.v_ref = self.leader_controller[0].low_pass_filter(
                    self.v_ref + v_rep + control_consensus*dt, self.v_ref, alpha=0.2
                )      
                self.altitude_ref = self.leader_controller[0].low_pass_filter(
                    self.altitude_ref + self.v_ref[:, 2] * dt, self.altitude_ref, alpha=0.2
                )

                if (self.path_tracking.goal_flag) and (e_bearing_norm < self.e_bearing_tol):
                    self.operation_mode = "formation_rectangle"
                    self.altitude_ref = pos_full[:, 2]
                    self.formation_controller._init_states("rectangle")

            case "formation_rectangle":
                control_consensus = self.formation_controller.consensus_law(pos_full)
                self.v_ref = control_consensus + v_rep       
                self.altitude_ref += self.v_ref[:, 2] * dt

                if (self.path_tracking.goal_flag) and (e_bearing_norm < self.e_bearing_tol):
                    self.operation_mode = "landing"

            case "landing":
                self.v_ref = np.zeros_like(pos_full)
                self.altitude_ref = np.zeros(n_drones)


        # --- Update camera once per timestep ---
        sim.cam.lookat = pos_full.mean(axis=0)
        # sim.cam.azimuth += 0.1

        

        # --- Compute individual drone control ---
        # print(np.array(data.sensor(f'pos_{0}').data))
        for i in range(n_drones):
            # Compute velocity-based control  
            u = self.controllers[i].vel_control_algorithm(
                state_full[i, :],
                self.v_ref[i, :2],      # full 3D velocity reference
                self.altitude_ref[i]
            )

            if i == 0 and self.operation_mode == "cruise":
                pos_ref = self.path_tracking.look_ahead_point(pos_full[0, :])    
                u = self.leader_controller[0].pos_control_algorithm(state_full[i, :], pos_ref+ v_rep[i, :]*dt)

            # Leader 1 drone follows the leader 0 to form the formation_scale
            if i == 2 and self.operation_mode == "cruise":
                ref_dir = (self.formation_controller.X_ref[i] - self.formation_controller.X_ref[0])
                pos_ref_2 = pos_full[0, :] + (self.formation_scale * ref_dir)
                u = self.leader_controller[1].pos_control_algorithm(state_full[i, :], pos_ref_2 + linvel_full[i, :]*dt)
                


            # Apply control to actuators
            for j in range(4):
                data.actuator(f"thrust{j+1}_{i}").ctrl = u[j]

        # --- Update previous time ---
        self.t_prev = sim.data.time

