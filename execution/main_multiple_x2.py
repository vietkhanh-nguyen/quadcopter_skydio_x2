import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from controls.quadcopter_controller import QuadcopterPIDController
from controls.consensus_controller import MultiAgentConsensus
from scipy.spatial.transform import Rotation 
import matplotlib.pyplot as plt
from scripts.multiple_drone_generate import save_multi_drone_xml

class MujocoSim:

    def __init__(self, xml_name, num_drones, simulation_time, fps):

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
        self.frames = None
        self.state = None
        self.temp = None

        # Print camera config
        self.print_camera_config = True #set to True to print camera config
                                         #this is useful for initializing view of the model)

        # Receive key input
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

        self.counter = 0
        self.num_drones = num_drones

        self.controllers = []
        self.formation_controller = MultiAgentConsensus(self.num_drones, K=.1)


    def set_up_ui(self):

        def keyboard(window, key, scancode, act, mods):
            if act == glfw.PRESS:
                if key == glfw.KEY_BACKSPACE:
                    mj.mj_resetData(self.model, self.data)
                    mj.mj_forward(self.model, self.data)
                elif key == glfw.KEY_ESCAPE:
                    glfw.set_window_should_close(window, True)

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
        self.cam.azimuth = -0.87
        self.cam.elevation = -25
        self.cam.distance =  12
        self.cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

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
            for i in range(self.num_drones):
                controller = QuadcopterPIDController(self.time_step)
                self.controllers.append(controller)

            pass


        def controller(model, data):
            #put the controller here. This function is called inside the simulation.]\
            height_ref = 4

            X = np.zeros((self.num_drones, 3))
            start_formation_control = True
            for id in range(self.num_drones):
                pos = np.array(self.data.sensor(f'pos_{id}').data)
                X[id, :] = pos
                if np.abs(pos[2] - height_ref) > 0.5:
                    start_formation_control = False

            self.cam.lookat = X.mean(axis=0)
            print(self.cam.lookat)
            if start_formation_control:
                control_consensus = self.formation_controller.consensus_law(X)
            else:
                control_consensus = np.zeros((self.num_drones, 2))

            for id in range(self.num_drones):
                body_pos = np.array(self.data.sensor(f'pos_{id}').data)
                body_quat = np.array(self.data.sensor(f'quat_{id}').data)
                body_linvel = self.data.sensor(f'vel_{id}').data
                body_angvel = self.data.sensor(f'gyro_{id}').data
                vel = np.hstack((body_linvel, body_angvel))
                euler = self.quat2euler(body_quat)
                state = np.concatenate([body_pos, euler, vel])
                # print(control_consensus[id, :])
                control_input = self.controllers[id].vel_control_algorithm(state, control_consensus[id, :2], height_ref)
                for j in range(1, 5):
                    actuator_name = f"thrust{j}_{id}"
                    self.data.actuator(actuator_name).ctrl = control_input[j-1]
            

            



        #initialize the controller
        init_controller(self.model,self.data)

        #set the controller
        mj.set_mjcb_control(controller)

        while not glfw.window_should_close(self.window):
            time_prev = self.data.time

            while (self.data.time - time_prev < 1.0/60.0):
                self.counter += 1
                mj.mj_step(self.model, self.data)

            if (self.data.time>=self.simulation_time):
                # X = np.zeros((self.num_drones, 2))
                # for id in range(self.num_drones):
                #     body_pos = np.array(self.data.sensor(f'pos_{id}').data)
                #     X[id, :] = body_pos[:2]
                
                # fig, ax = plt.subplots(figsize=(8, 6))
                # for i in range(self.num_drones):
                #     ax.scatter(X[i, 0], X[i, 1])
                # ax.set_aspect('equal', 'box')
                # plt.tight_layout()
                # plt.show()
                
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

if __name__ == "__main__":
    xml_path = '../mjcf/scene_multiple_x2.xml'
    simulation_time = 20 #simulation time
    num_drones = 8
    save_multi_drone_xml("mjcf/multiple_x2.xml", num_drones=num_drones)
    sim = MujocoSim(xml_path, num_drones, simulation_time, fps=60)
    sim.main_loop()









