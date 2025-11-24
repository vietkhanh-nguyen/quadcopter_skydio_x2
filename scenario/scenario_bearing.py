import numpy as np
from controls.quadcopter_controller import QuadcopterPIDController
from controls.consensus_controller import MultiAgentConsensus
from controls.pure_pursuit import PurePursuit
from path_planning.a_star_search import path_finding

class ScenarioBearingbasedConsensus:

    def init(self, sim):
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

    def update(self, sim, model, data):

        X = np.zeros((sim.num_drones, 3))
        for i in range(sim.num_drones):
            pos = np.array(data.sensor(f'pos_{i}').data)
            X[i, :] = pos
            if abs(pos[2] - self.altitude_ref[i]) < 0.05:
                self.tracking_flag = True

        sim.cam.lookat = X.mean(axis=0)

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

            # drone 0 follow path
            if i == 0 and self.tracking_flag:
                pos_ref = self.path_tracking.look_ahead_point(pos)
                u += 0.0 * self.follow_controller.pos_control_algorithm(state, pos_ref)

            for j in range(4):
                data.actuator(f"thrust{j+1}_{i}").ctrl = u[j]
