import os
import mujoco as mj
import numpy as np
import imageio



class MujocoRender:

    def __init__(self, xml_name, num_drones, simulation_time, time_step, fps, scenario):

        #get the full path
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_name)

        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(abspath)  # MuJoCo model
        if time_step is not None:
            self.model.opt.timestep = time_step
        self.time_step = self.model.opt.timestep
        self.data = mj.MjData(self.model)                # MuJoCo data
        self.cam = mj.MjvCamera()                        # Abstract camera
        self.opt = mj.MjvOption()                        # visualization options

        self.xml_path = abspath
        self.simulation_time = simulation_time
        self.fps = fps
        self.frames = []

        # Print camera config
        self.print_camera_config = False #set to True to print camera config
                                         #this is useful for initializing view of the model)

        # Receive key input
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
        
        self.num_drones = num_drones
        self.axs = None
        self.counter = 0

        self.scenario = scenario

    
    def main_loop(self):

        def init_controller(model, data):
            self.scenario.init(self)

        def controller(model, data):
            self.scenario.update(self, model, data)



        #initialize the controller
        init_controller(self.model,self.data)

        #set the controller
        mj.set_mjcb_control(controller)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        with mj.Renderer(self.model, 480, 480) as renderer:
            while self.data.time < self.simulation_time:
                self.counter +=1
                mj.mj_step(self.model, self.data)
                    

                if len(self.frames) < self.data.time * self.fps:
                    mj.mjv_updateScene(self.model, self.data, mj.MjvOption(),
                                    mj.MjvPerturb(), self.cam,
                                    mj.mjtCatBit.mjCAT_ALL, renderer.scene)
                    renderer.update_scene(self.data, self.cam)
                    pixels = renderer.render()
                    self.frames.append(pixels)

                if (self.data.time>=self.simulation_time ):
                    break


                #print camera configuration (help to initialize the view)
                if (self.print_camera_config):
                    print('self.cam.azimuth =',self.cam.azimuth,'\n','self.cam.elevation =',self.cam.elevation,'\n','self.cam.distance = ',self.cam.distance)
                    print('self.cam.lookat =np.array([',self.cam.lookat[0],',',self.cam.lookat[1],',',self.cam.lookat[2],'])')

                # Update scene and render
                mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                mj.mjtCatBit.mjCAT_ALL.value, self.scene)


            imageio.mimsave("outputs/mjc_render_output.mp4", self.frames, fps=self.fps)


if __name__ == "__main__":
    xml_path = '../mjcf/stewart_with_ball_force_ctrl.xml' #xml file (assumes this is in the same folder as this file)
    simulation_time = 30 #simulation time
    time_step = None
    sim = MujocoRender(xml_path, time_step, simulation_time, fps=60)
    sim.main_loop()

