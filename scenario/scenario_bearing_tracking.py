import numpy as np
from controls.quadcopter_controller import QuadcopterPIDController
from controls.consensus_controller import MultiAgentConsensus
from controls.pure_pursuit import PurePursuit
from path_planning.a_star_search import path_finding

class ScenarioBearingbasedTrackingConsensus:

    def __init__(self):
        self.name = "Drones Formation using Bearing-based Consensus Algorithm"

    def init(self, sim, model, data):

        sim.cam.azimuth = -0.87
        sim.cam.elevation = -25
        sim.cam.distance =  12
        sim.cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])
        
        path = path_finding()
        if path is None:
            path = sim.pos_ref

        self.path_tracking = PurePursuit(look_ahead_dist=2, waypoints=path)
        self.controllers = []
        self.formation_controller = MultiAgentConsensus(sim.num_drones, K=2, graph_type="complete")

        for _ in range(sim.num_drones):
            self.controllers.append(QuadcopterPIDController(sim.time_step))

        # PID for path following (để riêng)
        self.leader_controller = []
        for _ in range(2):
            self.leader_controller.append(QuadcopterPIDController(sim.time_step))

        self.tracking_flag = False
        self.formation_flag = False
        self.altitude_ref = 10 * np.ones(sim.num_drones)
        self.t_prev = sim.data.time
        self.v_ref = np.zeros((sim.num_drones, 3))

    def update(self, sim, model, data):
        n_drones = sim.num_drones
        dt = sim.data.time - self.t_prev

        # --- Read current states ---
        X = np.zeros((n_drones, 3))
        dX = np.zeros((n_drones, 3))
        ddX = np.zeros((n_drones, 3))
        for i in range(n_drones):
            # print(i)
            quat = np.array(data.sensor(f'quat_{i}').data)
            X[i, :] = np.array(data.sensor(f'pos_{i}').data)
            dX[i, :] = np.array(data.sensor(f'vel_{i}').data)
            body_acc = np.array(data.sensor(f'acc_{i}').data) 
            ddX[i, :] = self.controllers[i].linear_acc_world(quat, body_acc)
            if abs(X[i, 2] - self.altitude_ref[i]) < 0.05:
                self.formation_flag = True
        # print(ddX[0, :])
        # --- Check if all drones reached altitude reference ---
        # self.tracking_flag = np.any(np.abs(X[:, 2] - self.altitude_ref) < 0.5) and 
        e_bearing = self.formation_controller.compute_bearing_error(X)
        if np.linalg.norm(e_bearing) < 10:
            self.tracking_flag = True
            # self.tracking_flag = False
        # print(np.linalg.norm(e_bearing))
        # --- Update camera once per timestep ---
        sim.cam.lookat = X.mean(axis=0)
        # sim.cam.azimuth += 0.1

        # --- Compute formation control ---
        if self.tracking_flag:
            control_consensus = self.formation_controller.consensus_leader_vel_varying_law(
                X, dX, ddX, kp=100, kv=200
            )  # returns acceleration commands
            # --- Integrate reference velocity and altitude ---
            # self.v_ref += control_consensus*dt
            self.v_ref = self.controllers[0].low_pass_filter(self.v_ref + control_consensus*dt, self.v_ref, alpha=0.05)      
            self.altitude_ref += self.v_ref[:, 2] * dt
        elif self.formation_flag:
            control_consensus = self.formation_controller.consensus_law(X)
            self.v_ref = control_consensus          
            self.altitude_ref += self.v_ref[:, 2] * dt
        else:
            control_consensus = np.zeros_like(X)



        # --- Compute individual drone control ---
        # print(np.array(data.sensor(f'pos_{0}').data))
        for i in range(n_drones):
            pos = np.array(data.sensor(f'pos_{i}').data)
            quat = np.array(data.sensor(f'quat_{i}').data)
            linvel = np.array(data.sensor(f'vel_{i}').data)
            angvel = np.array(data.sensor(f'gyro_{i}').data)
            vel = np.hstack((linvel, angvel))
            state = np.concatenate([pos, quat, vel])

            # Compute velocity-based control

            u = self.controllers[i].vel_control_algorithm(
                state,
                self.v_ref[i, :2],      # full 3D velocity reference
                self.altitude_ref[i]
            )
            # Leader 1 drone follows the leader 0 to form the scale
            if i == 4 and self.tracking_flag:
                scale = 1.5
                pos_0 = np.array(data.sensor(f'pos_{0}').data)
                ref_dir = (self.formation_controller.X_ref[i] - self.formation_controller.X_ref[0])
                pos_ref_2 = pos_0 + (scale * ref_dir)
                self.leader_controller[1].Kp_pos = .01
                print(pos_ref_2)
                print(pos_0)
                u = self.leader_controller[1].pos_control_algorithm(state, pos_ref_2)
                
            if i == 0 and self.tracking_flag:
                # pos_ref = self.path_tracking.look_ahead_point(pos)
                # pos_ref = np.array([3, -2.5, 10])
                self.leader_controller[0].Kp_pos = .01
                u = self.leader_controller[0].pos_control_algorithm(state, pos)
                


            # Apply control to actuators
            for j in range(4):
                data.actuator(f"thrust{j+1}_{i}").ctrl = u[j]

        # --- Update previous time ---
        self.t_prev = sim.data.time

