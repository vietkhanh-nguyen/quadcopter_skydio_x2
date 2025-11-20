import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from controls.quadcopter_controller import QuadcopterPIDController
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt

class MujocoSim:

    def __init__(self, xml_name, simulation_time, fps):

        #get the full path
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_name)

        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(abspath)  # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data
        self.cam = mj.MjvCamera()                        # Abstract camera
        self.opt = mj.MjvOption()                        # visualization options

        self.xml_path = abspath
        self.simulation_time = simulation_time
        self.fps = fps
        self.time_step = self.model.opt.timestep

        # Tracking state
        self.tracked_data = None
        self.state = None
        self.temp = None

        # Print camera config
        self.print_camera_config = False #set to True to print camera config
                                         #this is useful for initializing view of the model)

        # Receive key input
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

        self.counter = 0

        self.vx_ref = 0.0
        self.vy_ref = 0.0 

        self.x_ref = 0.0
        self.y_ref = 0.0 
        self.z_ref = 0.0
        self.yaw_ref = 0.0

        self.keyboard_input_ref = np.zeros(5)
        self.pos_ref = np.zeros(3)

        self.controller = QuadcopterPIDController(self.time_step)

    def set_up_ui(self):

        def keyboard(window, key, scancode, act, mods):
            if act == glfw.PRESS:
                if key == glfw.KEY_BACKSPACE:
                    mj.mj_resetData(self.model, self.data)
                    mj.mj_forward(self.model, self.data)
                elif key == glfw.KEY_ESCAPE:
                    glfw.set_window_should_close(window, True)
                
            self.keyboard_control(window)

        def mouse_button(window, button, act, mods):
            self.button_left = (glfw.get_mouse_button(
                window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
            self.button_middle = (glfw.get_mouse_button(
                window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
            self.button_right = (glfw.get_mouse_button(
                window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

            # update mouse position
            glfw.get_cursor_pos(window)

        def mouse_move(window, xpos, ypos):
            dx = xpos - self.lastx
            dy = ypos - self.lasty
            self.lastx = xpos
            self.lasty = ypos

            # no buttons down: nothing to do
            if (not self.button_left) and (not self.button_middle) and (not self.button_right):
                return

            # get current window size
            width, height = glfw.get_window_size(window)

            # get shift key state
            PRESS_LEFT_SHIFT = glfw.get_key(
                window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            PRESS_RIGHT_SHIFT = glfw.get_key(
                window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

            # determine action based on mouse button
            if self.button_right:
                if mod_shift:
                    action = mj.mjtMouse.mjMOUSE_MOVE_H
                else:
                    action = mj.mjtMouse.mjMOUSE_MOVE_V
            elif self.button_left:
                if mod_shift:
                    action = mj.mjtMouse.mjMOUSE_ROTATE_H
                else:
                    action = mj.mjtMouse.mjMOUSE_ROTATE_V
            else:
                action = mj.mjtMouse.mjMOUSE_ZOOM

            mj.mjv_moveCamera(self.model, action, dx/height,
                            dy/height, self.scene, self.cam)

        def scroll(window, xoffset, yoffset):
            action = mj.mjtMouse.mjMOUSE_ZOOM
            mj.mjv_moveCamera(self.model, action, 0.0, 0.05 *
                            yoffset, self.scene, self.cam)
        
        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # 
        mj.mjv_defaultOption(self.opt)
        self.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True  # show joint frames
        self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True  # show contact points
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # install GLFW mouse and keyboard callbacks
        glfw.set_key_callback(self.window, keyboard)
        glfw.set_cursor_pos_callback(self.window, mouse_move)
        glfw.set_mouse_button_callback(self.window, mouse_button)
        glfw.set_scroll_callback(self.window, scroll)

        # Example on how to set camera configuration
        self.cam.azimuth = 0
        self.cam.elevation = -25
        self.cam.distance =  3
        self.cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

    def keyboard_control(self, window):
        step = 0.2
        angle_step = 0.1
        tol = 1e-2

        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        def approach_yaw(target):
            diff = normalize_angle(target - self.yaw_ref)
            if abs(diff) > tol:
                self.yaw_ref += np.clip(diff, -angle_step, angle_step)
                self.yaw_ref = normalize_angle(self.yaw_ref)

        max_speed = 1.5
        if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
            self.vy_ref = np.clip(self.vy_ref + step, -max_speed, max_speed)
            self.y_ref = self.y_ref + step
        elif glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
            self.vy_ref = np.clip(self.vy_ref - step, -max_speed, max_speed)
            self.y_ref = self.y_ref - step

        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
            self.vx_ref = np.clip(self.vx_ref + step, -max_speed, max_speed)
            self.x_ref = self.x_ref + step
        elif glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
            self.vx_ref = np.clip(self.vx_ref - step, -max_speed, max_speed)
            self.x_ref = self.x_ref - step

        if glfw.get_key(window, glfw.KEY_PERIOD) == glfw.PRESS:
            self.z_ref = np.clip(self.z_ref + step, 0, 3)
        elif glfw.get_key(window, glfw.KEY_COMMA) == glfw.PRESS:
            self.z_ref = np.clip(self.z_ref - step, 0, 3)

        if glfw.get_key(window, glfw.KEY_KP_0) == glfw.PRESS:
            self.vx_ref = 0
            self.vy_ref = 0

        # Yaw control targets
        yaw_targets = {
            glfw.KEY_Q: np.pi / 4,
            glfw.KEY_W: 0,
            glfw.KEY_E: - np.pi / 4,
            glfw.KEY_A: np.pi/2,
            glfw.KEY_D: -np.pi / 2,
            glfw.KEY_C: -3*np.pi / 4,
            glfw.KEY_X: np.pi,
            glfw.KEY_Z: 3 * np.pi / 4,
        }

        # Apply yaw control
        for key, target_yaw in yaw_targets.items():
            if glfw.get_key(window, key) == glfw.PRESS:
                approach_yaw(target_yaw)
                break  # Only act on one yaw key at a time

        self.keyboard_input_ref = np.array([self.x_ref, self.y_ref, self.z_ref, self.vx_ref, self.vy_ref])

    def quat2euler(self, quat_mujoco, degrees=False):
        # scipy quat = [x, y, z, constant]
        # mujoco quat = [constant, x, y, z]
        quat_scipy = np.array([quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]]) 
        r = Rotation.from_quat(quat_scipy)
        euler = r.as_euler('xyz', degrees=degrees)
        return euler
        
    def main_loop(self):
        self.set_up_ui()

        

        def init_controller(model,data):
            self.pos_ref = np.array([1.0, 1.0, 1.0])
            self.vel_ref = np.array([2.0, 2.0])

        def controller(model, data):
            #put the controller here. This function is called inside the simulation.]\
            
            # constant = 3.25
            # data.ctrl[:] = np.array([constant, constant, constant, constant])
            body_pos = np.array(data.sensor('body_pos').data)
            body_quat = np.array(data.sensor('body_quat').data)
            body_linvel = data.sensor('body_linvel').data
            body_angvel = data.sensor('body_gyro').data
            vel = np.hstack((body_linvel, body_angvel))
            euler = self.quat2euler(body_quat)
            self.state = np.concatenate([body_pos, euler, vel])

            
            # print(self.state)
            # self.data.ctrl = self.controller.pos_control_algorithm(self.state, keyboard_input_ref, self.yaw_ref)

            self.cam.lookat = body_pos 
            # self.cam.azimuth = np.degrees(euler[2])

            # Get position from data.sensordata
            # Each sensor contributes a fixed number of floats (3 for framepos)
            
            
            # print(body_pos)
            
            self.pos_ref = np.array([self.x_ref, self.y_ref, self.z_ref])
            # self.vel_ref = np.array([self.vx_ref, self.vy_ref])
            # self.data.ctrl = self.controller.pos_control_algorithm(self.state, self.keyboard_input_ref[:3], self.yaw_ref)
            self.data.ctrl = self.controller.vel_control_algorithm(self.state, self.keyboard_input_ref[3:5], self.keyboard_input_ref[2], self.yaw_ref)

            self.temp = euler
            self.counter += 1

        def tracking():
            if self.tracked_data is None:
                self.tracked_data = [[] for _ in range(2)]
            self.tracked_data[0].append(self.state)
            self.tracked_data[1].append(self.temp)

        def plot_sim(tracked_data):
            tracked_data0 = np.array(tracked_data[0])  # Shape: (timesteps, state_dim)
            tracked_data1 = np.array(tracked_data[1])  # Shape: (timesteps,)

            pitch = tracked_data0[:, 3]   # Pitch angle in radians or degrees
            pitch_ref = tracked_data1     # Reference pitch angle
            y = tracked_data0[:, 2]

            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            t = np.linspace(0, self.data.time, len(tracked_data0))
            # Subplot 1: Pitch angle tracking
            axs[0].plot(t, tracked_data0[:, 3], label='Roll', color='tab:orange', linewidth=2)
            axs[0].plot(t, tracked_data0[:, 4], label='Pitch', color='tab:olive', linestyle='--', linewidth=2)
            axs[0].plot(t, tracked_data0[:, 5], label='Yaw', color='tab:purple', linestyle='--', linewidth=2)
            axs[0].set_ylabel("Angle")
            axs[0].set_title("Angle Tracking Over Time")
            axs[0].legend()
            axs[0].grid(True)

            # Subplot 2: X position
            axs[1].plot(t, tracked_data0[:, 6], label='X Position', color='tab:red', linewidth=2)
            axs[1].plot(t, tracked_data0[:, 7], label='Y Position', color='tab:green', linewidth=2)
            axs[1].plot(t, tracked_data0[:, 2], label='Z Position', color='tab:blue', linewidth=2)
            axs[1].set_xlabel("Timestep")
            axs[1].set_ylabel("Position")
            axs[1].set_title("Position Tracking Over Time")
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout()
            plt.show()




        #initialize the controller
        init_controller(self.model,self.data)

        #set the controller
        mj.set_mjcb_control(controller)

        while not glfw.window_should_close(self.window):
            time_prev = self.data.time

            while (self.data.time - time_prev < 1.0/60.0):
                mj.mj_step(self.model, self.data)

            tracking()

            if (self.data.time>=self.simulation_time):
                break

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            #print camera configuration (help to initialize the view)
            if (self.print_camera_config):
                print('cam.azimuth =',self.cam.azimuth,';','cam.elevation =',self.cam.elevation,';','cam.distance = ',self.cam.distance)
                print('cam.lookat =np.array([',self.cam.lookat[0],',',self.cam.lookat[1],',',self.cam.lookat[2],'])')

            # Update scene and render
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                            mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()
        plot_sim(self.tracked_data)

if __name__ == "__main__":
    xml_path = '../mjcf/scene.xml' #xml file (assumes this is in the same folder as this file)
    simulation_time = 100 #simulation time

    sim = MujocoSim(xml_path, simulation_time, fps=60)
    sim.main_loop()









