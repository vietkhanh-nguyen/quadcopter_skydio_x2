import numpy as np
from controls.quadcopter_controller import QuadcopterPIDController
from controls.pure_pursuit import PurePursuit
from path_planning.a_star_search import path_finding

class ScenarioDroneTracking:
    """
    Scenario for single-drone altitude tracking + path following.
    """

    def __init__(self):
        self.name = "Drone Tracking Trajectory"

    def init(self, sim, model, data):
        """
        Initialize controller and tracking references.
        sim: MujocoSim object
        """

        sim.cam.azimuth = 45.0 
        sim.cam.elevation = -30.0
        sim.cam.distance =  8.0

        # Plan path
        path, env = path_finding()
        if path is None:
            path = sim.pos_ref  # fallback if path not found

        # Pure pursuit path tracker
        self.path_tracking = PurePursuit(look_ahead_dist=2, waypoints=path)

        # PID controller for the drone
        self.controller = QuadcopterPIDController(sim.time_step)

        # Tracking flags and altitude reference
        self.tracking_flag = False
        self.altitude_ref_init = 5.0
        self.yaw_ref = 0.0  # default yaw reference

        # Store state
        self.state = None

        

    def update(self, sim, model, data):
        """
        Called at each simulation step.
        sim: MujocoSim object
        model: mj.MjModel
        data: mj.MjData
        """
        # Read sensors
        body_pos = np.array(data.sensor('body_pos').data)
        body_quat = np.array(data.sensor('body_quat').data)
        body_linvel = np.array(data.sensor('body_linvel').data)
        body_angvel = np.array(data.sensor('body_gyro').data)

        # Compose full state [pos, quat, vel]
        vel = np.hstack((body_linvel, body_angvel))
        self.state = np.concatenate([body_pos, body_quat, vel])

        # Update camera to follow drone
        sim.cam.lookat = body_pos

        # Check if drone reached initial altitude
        if np.abs(body_pos[2] - self.altitude_ref_init) < 0.05:
            self.tracking_flag = True

        # Determine reference position
        if self.tracking_flag:
            pos_ref = self.path_tracking.look_ahead_point(body_pos)
            print(
                f"Pos_ref: {np.round(pos_ref, 2)}, "
                f"Body_pos: {np.round(body_pos, 2)}"
            )
        else:
            pos_ref = np.copy(body_pos)
            pos_ref[2] = self.altitude_ref_init

        # Compute control input
        control_input = self.controller.pos_control_algorithm(self.state, pos_ref, self.yaw_ref)

        # Apply control to actuators
        data.ctrl[:] = control_input
