import numpy as np
from controls.quadcopter_controller import QuadcopterPIDController
from controls.consensus_controller import MultiAgentConsensus
from controls.pure_pursuit import PurePursuit
from path_planning.a_star_search import path_finding
from utilities.gen_multi_agent_graph import *

class ScenarioBearingbasedConsensus:

    def __init__(self):
        self.name = "Drones Formation using Bearing-based Consensus Algorithm"

    def init(self, sim, model, data):

        sim.cam.azimuth = -0.87
        sim.cam.elevation = -25
        sim.cam.distance =  16
        sim.cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])
        
        path = path_finding()
        if path is None:
            path = sim.pos_ref

        self.path_tracking = PurePursuit(look_ahead_dist=2, waypoints=path)
        self.controllers = []
        self.formation_controller = MultiAgentConsensus(sim.num_drones, K=1)

        for _ in range(sim.num_drones):
            self.controllers.append(QuadcopterPIDController(sim.time_step))

        # PID for path following (để riêng)
        self.follow_controller = QuadcopterPIDController(sim.time_step)

        self.tracking_flag = False
        self.altitude_ref = 5 * np.ones(sim.num_drones)
        self.not_landing_flag = True

    def update(self, sim, model, data):

        X = np.zeros((sim.num_drones, 3))
        for i in range(sim.num_drones):
            pos = np.array(data.sensor(f'pos_{i}').data)
            X[i, :] = pos
            if abs(pos[2] - self.altitude_ref[i]) < 0.05:
                self.tracking_flag = True

        sim.cam.lookat = X.mean(axis=0)
        sim.cam.azimuth += 0.1

        if sim.data.time > 30 and self.not_landing_flag:
            self.not_landing_flag = False
            self.formation_controller.X_ref = gen_rectangle(sim.num_drones, spacing=2.0, agents_per_row=3)
            # X_ref = gen_sphere(self.n_agents, self.dim_state, radius=1)
            diff = self.formation_controller.X_ref[:, np.newaxis, :] - self.formation_controller.X_ref[np.newaxis, :, :]
            norm = np.linalg.norm(diff, axis=2, keepdims=True)
            norm = np.where(norm == 0, 1e-9, norm)   # avoid divide-by-zero
            self.formation_controller.dir_ref = -diff / norm

        e_bearing = self.formation_controller.compute_bearing_error(X)
        print(f"time: {sim.data.time:.2f}, e_bearing: {np.linalg.norm(e_bearing):.2f}")

        if (not self.not_landing_flag) and (np.linalg.norm(e_bearing) < 1):
            self.altitude_ref = np.zeros(sim.num_drones)

        if self.tracking_flag:
            control_consensus = self.formation_controller.consensus_law(X)
        else:
            control_consensus = np.zeros_like(X)

        for i in range(sim.num_drones):

            pos = np.array(data.sensor(f'pos_{i}').data)
            quat = np.array(data.sensor(f'quat_{i}').data)
            linvel = data.sensor(f'vel_{i}').data
            angvel = data.sensor(f'gyro_{i}').data
            vel = np.hstack((linvel, angvel))
            state = np.concatenate([pos, quat, vel])
            self.altitude_ref[i] += control_consensus[i, 2] * 0.01



            u = self.controllers[i].vel_control_algorithm(
                state,
                control_consensus[i, :2],
                self.altitude_ref[i],
            )

            for j in range(4):
                data.actuator(f"thrust{j+1}_{i}").ctrl = u[j]
