import numpy as np
from scipy.spatial.transform import Rotation 


class QuadcopterPIDController:

    def __init__(self, time_step):
        self.cal_de = True
        self.dt = time_step

        # Position Control PID (for roll & pitch references)
        self.Kp_pos = .1
        self.Ki_pos = 0
        self.Kd_pos = .35
        self.int_pos = np.array([0.0, 0.0])
        self.prev_e_pos = None
        self.prev_de_pos = None

        # Velocity Control PID (for roll & pitch references)
        self.Kp_vel = .1
        self.Ki_vel = 0
        self.Kd_vel = 0
        self.int_vel = np.array([0.0, 0.0])
        self.prev_e_vel = None
        self.prev_de_vel = None

        # Thrust PID
        self.vehicle_mass_offset = 3.25
        self.Kp_t = 2
        self.Ki_t = 0
        self.Kd_t = 2
        self.int_t = 0
        self.prev_e_t = None
        self.prev_de_t = None

        # Roll PID
        self.Kp_r = 5
        self.Ki_r = 0
        self.Kd_r = 2.5
        self.int_r = 0
        self.prev_e_r = None
        self.prev_de_r = None

        # Pitch PID
        self.Kp_p = 5
        self.Ki_p = 0
        self.Kd_p = 2.5
        self.int_p = 0
        self.prev_e_p = None
        self.prev_de_p = None

        # Yaw PID
        self.Kp_y = 1
        self.Ki_y = 0
        self.Kd_y = 10
        self.int_y = 0
        self.prev_e_y = None
        self.prev_de_y = None
  
    def quat2euler(self, quat_mujoco, degrees=False):
        # scipy quat = [x, y, z, constant]
        # mujoco quat = [constant, x, y, z]
        quat_scipy = np.array([quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]]) 
        r = Rotation.from_quat(quat_scipy)
        euler = r.as_euler('xyz', degrees=degrees)
        return euler
    
    def rpy_to_rotmat(self, euler_angle):
        roll, pitch, yaw = euler_angle
        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        Rx = np.array([[1, 0, 0],
                    [0, cr, -sr],
                    [0, sr, cr]])

        Ry = np.array([[cp, 0, sp],
                    [0, 1, 0],
                    [-sp, 0, cp]])

        Rz = np.array([[cy, -sy, 0],
                    [sy,  cy, 0],
                    [0,   0,  1]])

        # World ← Body
        R = Rz @ Ry @ Rx
        return R
    
    def linear_acc_world(self, body_quat, body_acc, gravity=np.array([0, 0, -9.81])):
        euler_angle = self.quat2euler(body_quat)
        rot_matrix = self.rpy_to_rotmat(euler_angle)
        acc_world = rot_matrix@body_acc
        linear_acc_world = acc_world + gravity
        return linear_acc_world

    
    def normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def low_pass_filter(self, raw_de, pre_de, alpha = 0.1):
        return (1 - alpha)*pre_de + alpha*raw_de

    def roll_controller(self, e):
        e = self.normalize_angle(e)
        if self.prev_e_r is None:
            self.prev_e_r = e
            self.prev_de_r = np.zeros_like(e)
        de = (e - self.prev_e_r) / self.dt
        self.int_r += e * self.dt
        self.prev_e_r = e
        self.prev_de_r = de
        return self.Kp_r * e + self.Ki_r * self.int_r + self.Kd_r * de

    def pitch_controller(self, e):
        e = self.normalize_angle(e)
        if self.prev_e_p is None:
            self.prev_e_p = e
            self.prev_de_p = np.zeros_like(e)
        de = (e - self.prev_e_p) / self.dt
        self.int_p += e * self.dt
        self.prev_e_p = e
        self.prev_de_p = de
        return self.Kp_p * e + self.Ki_p * self.int_p + self.Kd_p * de

    def yaw_controller(self, e):
        e = self.normalize_angle(e)
        if self.prev_e_y is None:
            self.prev_e_y = e
            self.prev_de_y = np.zeros_like(e)
        raw_de = (e - self.prev_e_y) / self.dt
        de = self.low_pass_filter(raw_de, self.prev_de_y)
        self.int_y += e * self.dt
        self.prev_e_y = e
        self.prev_de_y = de
        return self.Kp_y * e + self.Ki_y * self.int_y + self.Kd_y * de

    def thrust_controller(self, e, de=None):
        if self.prev_e_t is None:
            self.prev_e_t = e
            self.prev_de_t = np.zeros_like(e)
        if de is None:
            raw_de = (e - self.prev_e_t) / self.dt
            de = self.low_pass_filter(raw_de, self.prev_de_t)
        self.int_t += e * self.dt
        self.prev_e_t = e
        self.prev_de_t = de
        return self.vehicle_mass_offset + self.Kp_t * e + self.Ki_t * self.int_t + self.Kd_t * de

    def position_controller(self, e, de=None):
        e[1] = -e[1]
        de[1] = -de[1]
        if self.prev_e_pos is None:
            self.prev_e_pos = e
            self.prev_de_pos = np.zeros_like(e)

        if de is None:
            raw_de = (e - self.prev_e_pos) / self.dt
            de = self.low_pass_filter(raw_de, self.prev_de_pos)

        self.int_pos += e * self.dt
        self.prev_e_pos = e
        self.prev_de_pos = de
        out = self.Kp_pos * e + self.Ki_pos * self.int_pos + self.Kd_pos * de
        return out  # [pitch_ref, roll_ref]
    
    def velocity_controller(self, e):
        e[1] = -e[1]
        if self.prev_e_vel is None:
            self.prev_e_vel = e
            self.prev_de_vel = np.zeros_like(e)
        raw_de = (e - self.prev_e_vel) / self.dt
        de = self.low_pass_filter(raw_de, self.prev_de_vel, alpha=0.3)
        self.int_vel += e * self.dt
        self.prev_e_vel = e
        self.prev_de_vel = de
        out = self.Kp_vel * e + self.Ki_vel * self.int_vel + self.Kd_vel * de
        return out  # [pitch_ref, roll_ref]

    def motor_mixing_algorithm(self, t, r, p, y):
        m1 = t - r + p - y
        m2 = t + r + p + y
        m3 = t + r - p - y
        m4 = t - r - p + y
        return m1, m2, m3, m4

    def pos_control_algorithm(self, state, pos_ref, yaw_ref=None):
        # Read system state
        state = np.array(state)
        x, y, z, quat_cons, quat_x, quat_y, quat_z, vx, vy, vz, v_roll, v_pitch, vyaw = state
        roll, pitch, yaw = self.quat2euler(np.array([quat_cons, quat_x, quat_y, quat_z]))
        x_ref, y_ref, altitude_ref = pos_ref.flatten()
        position_ref = np.array([x_ref, y_ref])

        # Generate roll and pitch reference from position error
        pos_error_world = position_ref - np.array([x, y])
        # yaw = self.normalize_angle(yaw)
        R_world_to_body = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        pos_error_body = R_world_to_body @ pos_error_world.reshape(2, 1)
        vel_world = np.array([vx, vy])
        vel_body = R_world_to_body @ vel_world.reshape(2, 1)
        pitch_ref, roll_ref  = self.position_controller(pos_error_body.reshape(-1), -vel_body.reshape(-1))
        u_thrust = self.thrust_controller(altitude_ref - z, -vz)
        u_roll = self.roll_controller(roll_ref - roll)
        u_pitch = self.pitch_controller(pitch_ref - pitch)
        u_yaw = self.yaw_controller(yaw_ref-yaw) if yaw_ref is not None else self.yaw_controller(-yaw)
        return np.array(self.motor_mixing_algorithm(u_thrust, u_roll, u_pitch, u_yaw))
    

    def vel_control_algorithm(self, state, vel_ref, altitude_ref, yaw_ref=None):
        # Read system state
        state = np.array(state)
        x, y, z, quat_cons, quat_x, quat_y, quat_z, vx, vy, vz, v_roll, v_pitch, vyaw = state
        roll, pitch, yaw = self.quat2euler(np.array([quat_cons, quat_x, quat_y, quat_z]))

        # Generate roll and pitch reference from position error
        vel_error_world = vel_ref - np.array([vx, vy])
        yaw = self.normalize_angle(yaw)
        R_world_to_body = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        vel_error_body = R_world_to_body @ vel_error_world
        pitch_ref, roll_ref = self.velocity_controller(vel_error_body)
        u_thrust = self.thrust_controller(altitude_ref - z, -vz)
        u_roll = self.roll_controller(roll_ref - roll)
        u_pitch = self.pitch_controller(pitch_ref - pitch)
        u_yaw = self.yaw_controller(self.normalize_angle(yaw_ref-yaw)) if yaw_ref is not None else self.yaw_controller(self.normalize_angle(-yaw))
        return np.array(self.motor_mixing_algorithm(u_thrust, u_roll, u_pitch, u_yaw))

    def vel_body_control_algorithm(self, state, vel_ref, altitude_ref, yaw_ref=None):
        # Read system state
        state = np.array(state)
        x, y, z, quat_cons, quat_x, quat_y, quat_z, vx, vy, vz, v_roll, v_pitch, vyaw = state
        roll, pitch, yaw = self.quat2euler(np.array([quat_cons, quat_x, quat_y, quat_z]))

        yaw_norm = self.normalize_angle(yaw)
        R_world_to_body = np.array([
            [np.cos(yaw_norm),  np.sin(yaw_norm)],
            [-np.sin(yaw_norm), np.cos(yaw_norm)]
        ])
        vel_body_curr = R_world_to_body @ np.array([vx, vy])
        vel_error = vel_ref - vel_body_curr
        yaw = self.normalize_angle(yaw)
        pitch_ref, roll_ref = self.velocity_controller(vel_error)
        u_thrust = self.thrust_controller(altitude_ref - z, -vz)
        u_roll = self.roll_controller(roll_ref - roll)
        u_pitch = self.pitch_controller(pitch_ref - pitch)
        u_yaw = self.yaw_controller(self.normalize_angle(yaw_ref-yaw)) if yaw_ref is not None else self.yaw_controller(self.normalize_angle(-yaw))
        return np.array(self.motor_mixing_algorithm(u_thrust, u_roll, u_pitch, u_yaw))
