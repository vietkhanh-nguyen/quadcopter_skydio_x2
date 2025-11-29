import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 
from utilities.gen_multi_agent_graph import build_universal_rigid_graph, gen_icosahedron, gen_sphere, gen_rectangle


import numpy as np

class MultiAgentConsensus:
    def __init__(self, n_agents=7, K=1, graph_type="complete", center_virtual_agent=False):
        self.n_agents = n_agents
        self.K = K/self.n_agents
        self.impact_dis = 1
        self.muy = (1 + self.impact_dis**4)/self.impact_dis**4
        # Initialize system matrices
        self.graph_type = graph_type
        self.dim_state = 3
        self.n_virtual_agent = 2 if center_virtual_agent else 0
        self._init_matrices()
        self._init_states()

    # ----------------------------------------------------------------------
    def _init_matrices(self):
        """Initialize Laplacian and dynamic matrices."""
        if self.graph_type == "complete":
            A_graph = np.ones((self.n_agents+self.n_virtual_agent , self.n_agents+self.n_virtual_agent ), dtype=int)
            np.fill_diagonal(A_graph, 0)
            self.num_edges = int(self.n_agents*(self.n_agents - 1)/2)
        elif self.graph_type == "universal_rigid":
            num_edges, A_graph = build_universal_rigid_graph(self.n_agents, self.dim_state)
            self.num_edges = num_edges
        else:
            raise ValueError("graph_type must be 'complete' or 'universal_rigid'.")
        # Ma trận Laplacian
        self.A_graph = A_graph

    # ----------------------------------------------------------------------
    def _init_states(self, formation=None):

        # state now includes (x, y, z)
        if formation == None:
            if self.n_agents != 12:
                X_ref = gen_sphere(self.n_agents, self.dim_state, radius=1)
            else:
                X_ref = gen_icosahedron(radius=5)
        else:
            match formation:
                case "icosahedron":
                    X_ref = gen_icosahedron(radius=3)
                case  "sphere":
                    X_ref = gen_sphere(self.n_agents, self.dim_state, radius=1)
                case "rectangle":
                    X_ref = gen_rectangle(self.n_agents, spacing=0.8, agents_per_row=3)

        # X_ref = gen_sphere(self.n_agents, self.dim_state, radius=1)
        diff = X_ref[:, np.newaxis, :] - X_ref[np.newaxis, :, :]
        norm = np.linalg.norm(diff, axis=2, keepdims=True)
        norm = np.where(norm == 0, 1e-9, norm)   # avoid divide-by-zero
        self.dir_ref = -diff / norm
        self.X_ref = X_ref
        

    def projection_matrix(self, gij):
        gij = np.asarray(gij).reshape(-1)
        norm = np.linalg.norm(gij)
        if norm == 0:
            return np.eye(self.dim_state)  # or np.zeros((d,d)) depending on desired behavior
        g = (gij / norm).reshape(-1,1)
        return np.eye(self.dim_state) - g @ g.T
    
    # ----------------------------------------------------------------------
    def consensus_law(self, X):
        diff_matrix = X[:, np.newaxis, :] - X[np.newaxis, :, :] # diff[i, j, :] = X[i, :] - X[j, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2, keepdims=True)
        dist_matrix = np.where(dist_matrix == 0, 1e-9, dist_matrix)   # avoid divide-by-zero
        dir_cur = -diff_matrix / dist_matrix
        # dist_matrix = dist_matrix.reshape(self.n_agents, self.n_agents)

        u = np.zeros((self.n_agents, self.dim_state)) 
        for i in range(self.n_agents):
            for j in range(self.n_agents):

                e = dist_matrix[i, j]**2 - self.impact_dis**2
                if e <= 0 and i != j:
                    dBij = (-4*self.muy*e)/(1 + e**2)**2 * diff_matrix[i, j, :]
                    u[i] += 2*dBij / (1 - self.muy*(e**2 / (1 + e**2)))

                if self.A_graph[i, j] == 0:
                    continue  
                u[i] -= self.projection_matrix(dir_cur[i, j, :])@self.dir_ref[i, j, :]

        return self.K*u
    
    def consensus_leader_vel_varying_law(self, X, dX, ddX, kp=1, kv=2, ka=3, kadv=1):

        diff_matrix = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2, keepdims=True)
        dist_matrix = np.where(dist_matrix == 0, 1e-9, dist_matrix)   # avoid divide-by-zero
        dist_matrix = dist_matrix.reshape(self.n_agents + self.n_virtual_agent , self.n_agents + self.n_virtual_agent)

        u = np.zeros((self.n_agents, self.dim_state)) 
        K_mat = np.zeros((self.n_agents, self.dim_state, self.dim_state)) 

        for i in range(self.n_agents):
            u_avoid_obs = np.zeros(self.dim_state)
            for j in range(self.n_agents + self.n_virtual_agent):
                e = dist_matrix[i, j]**2 - self.impact_dis**2
                if e <= 0 and i != j and j!=self.n_agents:
                    dBij = (-4*self.muy*e)/(1 + e**2)**2 * diff_matrix[i, j, :]
                    u_avoid_obs += dBij / (1 - self.muy*(e**2 / (1 + e**2)))

                if self.A_graph[i, j] == 0:
                    continue  
                Pgij_ref = self.projection_matrix(self.dir_ref[i, j, :])
                ep = X[i, :] - X[j, :]
                norm = np.linalg.norm(ep)
                # Choose vector based on norm
                if norm > 1:
                    vec = ep
                else:
                    if norm > 1e-12:  # avoid division by zero
                        vec = ep / norm
                    else:
                        vec = np.zeros_like(ep)  # if X[i]==X[j]
                u[i] += Pgij_ref@(kp*vec + kv*(dX[i, :] - dX[j, :]) - ka*ddX[j, :])
                K_mat[i, :, :] += Pgij_ref
            u[i] = -np.linalg.pinv(K_mat[i, :, :]) @ u[i] + kadv*u_avoid_obs
        return self.K*u
    
    def avoid_collision_law(self, dist_matrix, X, i):
        dB = np.zeros(self.dim_state)
        for j in range(self.n_agents):
            if j == i:
                continue
            dij = dist_matrix[i, j]
            e = dij**2 - self.impact_dis**2
            muy = (1 + self.impact_dis**4)/self.impact_dis**4
            if e <= 0:
                dBij = (-4*muy*e)/(1 + e**2)**2 * (X[i, :] - X[j, :])
                dB += dBij / (1 - muy*(e**2 / (1 + e**2)))
        return dB

    def compute_bearing_error(self, X):
        """
        Compute bearing errors for all agents.
        
        Args:
            X: (n_agents, dim_state) array of positions
            dir_ref: (n_agents, n_agents, dim_state) array of reference bearings g^*_ij
        
        Returns:
            e_bearing: (n_agents, n_agents, dim_state) array of bearing errors
                        e_bearing[i, j, :] = P_{g^*_ij} @ (X[i] - X[j])
        """
        e_bearing = np.zeros((self.n_agents, self.n_agents, self.dim_state))
        
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue  # skip self
                e_bearing[i, j, :] = self.projection_matrix(self.dir_ref[i, j, :]) @ (X[i] - X[j])
        
        return e_bearing


    # ----------------------------------------------------------------------
    def plot(self):
        """Plot the 3D consensus evolution for all agents."""
        if self.dim_state != 3:
            raise ValueError("3D plotting requires dim_state = 3.")

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectories
        for i in range(self.n_agents):
            ax.plot(
                self.X[i, 0, :],
                self.X[i, 1, :],
                self.X[i, 2, :],
                label=f"Agent {i+1}"
            )
            # Mark final point
            ax.scatter(
                self.X[i, 0, -1],
                self.X[i, 1, -1],
                self.X[i, 2, -1],
                s=40
            )

        # Labels and formatting
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title("3D Consensus Trajectories of All Agents")

        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True)
        ax.set_box_aspect([1, 1, 1])  # equal aspect ratio

        plt.tight_layout()
        plt.show()

    def plot_gif_3d(self, T, dt, max_frames=100):
        """Animate and export the consensus evolution in 3D for all agents."""
        from mpl_toolkits.mplot3d import Axes3D
        self.n_steps = int(T / dt)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Colors
        colors = plt.cm.tab10.colors
        if self.n_agents > len(colors):
            colors = plt.cm.tab20.colors

        # Initialize lines and points
        lines = []
        points = []
        for i in range(self.n_agents):
            color = colors[i % len(colors)]
            (line,) = ax.plot([], [], [], lw=1.5, color=color)
            (point,) = ax.plot([], [], [], "o", markersize=4, color=color)
            lines.append(line)
            points.append(point)

        # Axis limits
        x_min, x_max = np.min(self.X[:, 0, :]), np.max(self.X[:, 0, :])
        y_min, y_max = np.min(self.X[:, 1, :]), np.max(self.X[:, 1, :])
        z_min, z_max = np.min(self.X[:, 2, :]), np.max(self.X[:, 2, :])

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_box_aspect([1,1,1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Consensus evolution")
        ax.grid(True)

        def init():
            for line, point in zip(lines, points):
                line.set_data([], [])
                line.set_3d_properties([])
                point.set_data([], [])
                point.set_3d_properties([])
            return lines + points

        def update(frame):
            for i in range(self.n_agents):
                start = max(frame - 20, 0)
                x = self.X[i, 0, start:frame]
                y = self.X[i, 1, start:frame]
                z = self.X[i, 2, start:frame]
                if len(x) == 0:
                    continue
                lines[i].set_data(x, y)
                lines[i].set_3d_properties(z)
                points[i].set_data([x[-1]], [y[-1]])
                points[i].set_3d_properties([z[-1]])
            return lines + points

        # Subsample frames
        skip = max(1, int(self.n_steps / max_frames))
        frames = range(0, self.n_steps, skip)

        anim = animation.FuncAnimation(fig, update, init_func=init,
                                    frames=frames, interval=50, blit=True)

        filename = "outputs/consensus_evolution_3d.gif"
        writer = animation.PillowWriter(fps=20)
        anim.save(filename, writer=writer)
        print(f"✅ 3D GIF saved as {filename}")

        plt.close(fig)


    def plot_gif(self, T, dt, max_frames=100):
        """Animate and export the consensus evolution in the x1–x2 plane for all agents."""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Choose colors for each agent
        colors = plt.cm.tab10.colors  # 10 distinct colors
        if self.n_agents > len(colors):
            colors = plt.cm.tab20.colors  # fallback for more agents

        # Initialize lines and points with the same color per agent
        lines = []
        points = []
        for i in range(self.n_agents):
            color = colors[i % len(colors)]
            (line,) = ax.plot([], [], lw=1.5, color=color)
            (point,) = ax.plot([], [], "o", markersize=4, color=color)
            lines.append(line)
            points.append(point)

        # Axis limits and 1:1 aspect ratio
        ax.set_xlim(np.min(self.X[::self.dim_state, :]) - 0.5, np.max(self.X[::self.dim_state, :]) + 0.5)
        ax.set_ylim(np.min(self.X[1::self.dim_state, :]) - 0.5, np.max(self.X[1::self.dim_state, :]) + 0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x1 (position)")
        ax.set_ylabel("x2 (position)")
        ax.set_title("Consensus evolution of all agents in state space")
        ax.grid(True)
        # ax.legend([f"Agent {i+1}" for i in range(self.n_agents)], loc="upper right", ncol=3, fontsize=8)

        def init():
            for line, point in zip(lines, points):
                line.set_data([], [])
                point.set_data([], [])
            return lines + points

        def update(frame):
            for i in range(self.n_agents):
                x = self.X[i, 0, np.max([(frame-20), 0]):frame]
                y = self.X[i, 1, np.max([(frame-20), 0]):frame]
                if len(x) == 0 or len(y) == 0:
                    continue
                lines[i].set_data(x, y)
                points[i].set_data([x[-1]], [y[-1]])  # same color as line
            return lines + points

        # Subsample frames to have at most max_frames
        skip = max(1, int(self.n_steps / max_frames))
        frames = range(0, self.n_steps, skip)

        anim = animation.FuncAnimation(
            fig, update, init_func=init, frames=frames, interval=30, blit=True
        )

        filename = "outputs/consensus_evolution.gif"
        writer = animation.PillowWriter(fps=30)
        anim.save(filename, writer=writer)
        print(f"✅ GIF saved as {filename}")

        plt.close(fig)


    def plot_Xref(self):
        """
        Visualize the reference positions X_ref on a 3D sphere.
        """
        if self.dim_state != 3:
            raise ValueError("X_ref visualization only applies to 3D sphere mode.")

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Extract points
        Xr = self.X_ref.reshape(self.n_agents, 3)
        xs, ys, zs = Xr[:, 0], Xr[:, 1], Xr[:, 2]

        # Draw sphere for visualization
        radius = np.linalg.norm(Xr[0])  # should equal 0.5
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        xsph = radius * np.outer(np.cos(u), np.sin(v))
        ysph = radius * np.outer(np.sin(u), np.sin(v))
        zsph = radius * np.outer(np.ones_like(u), np.cos(v))

        ax.plot_surface(xsph, ysph, zsph, alpha=0.15, color="blue", linewidth=0)

        # Reference nodes
        ax.scatter(xs, ys, zs, color='red', s=40, label="Reference points")

        # Label points
        for i in range(self.n_agents):
            ax.text(xs[i], ys[i], zs[i], f"{i}", fontsize=9)

        ax.set_title("Reference configuration on a 3D sphere")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1,1,1])
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()






# ----------------------------------------------------------------------
# Example test
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Example test / quick validation
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import numpy as np

    # Simulation parameters
    n_agents = 8           # total agents (including 2 leaders)
    dim_state = 3
    K = 1
    dt = 0.02
    T = 200.0               # total sim time (seconds)
    kp = 10.0
    kv = 20.0

    # Create simulator (your MultiAgentConsensus class must be in scope)
    sim = MultiAgentConsensus(
        n_agents=n_agents,
        K=K,
        graph_type="complete",
        dim_state=dim_state
    )

    # --- Initial Condition ---
    # start agents near their reference positions with a small random perturbation
    np.random.seed(12)
    x0 = sim.X_ref + 0.06 * np.random.randn(n_agents, dim_state)

    # Ensure leaders start at their commanded positions (override for agents 0 and 1)
    def leader0_position(t):
        # translation leader: circular along x-axis (only x moves)
        return np.array([
            1.5 * np.cos(0.4 * t),
            0.9 * np.sin(0.2 * t),
            0.2 * np.sin(0.15 * t)
        ])

    def leader0_velocity(t):
        return np.array([
            -1.5 * 0.4 * np.sin(0.4 * t),
            0.9 * 0.2 * np.cos(0.2 * t),
            0.2 * 0.15 * np.cos(0.15 * t)
        ])

    def leader0_acceleration(t):
        return np.array([
            -1.5 * (0.4**2) * np.cos(0.4 * t),
            -0.9 * (0.2**2) * np.sin(0.2 * t),
            -0.2 * (0.15**2) * np.sin(0.15 * t)
        ])

    # Leader 1 controls scale by moving on the x-axis relative to leader 0
    # base distance r01_ref is the reference distance between ref nodes 0 and 1
    r01_ref = np.linalg.norm(sim.X_ref[1] - sim.X_ref[0])

    # We'll make leader1 vary its distance from leader0 to command scale s(t)
    def leader1_scale_and_derivs(t):
        # scale s(t) = 1 + 0.4*sin(0.08 t)  (example)
        s = 1.0 
        sd = 0
        sdd = 0
        return s, sd, sdd

    def leader1_position(t):
        # place leader1 on x-axis at distance s(t)*r01_ref from leader0
        s, _, _ = leader1_scale_and_derivs(t)
        p0 = leader0_position(t)
        # direction from leader0->ref1 (in reference configuration)
        ref_dir = sim.X_ref[1] - sim.X_ref[0]
        ref_dir_norm = np.linalg.norm(ref_dir)
        if ref_dir_norm < 1e-9:
            ref_unit = np.array([1.0, 0.0, 0.0])
        else:
            ref_unit = ref_dir / ref_dir_norm
        return p0 + (s * r01_ref) * ref_unit

    def leader1_velocity(t):
        # v1 = v0 + s_dot * r01_ref * ref_unit  (approx, ignoring ref_unit rotation)
        s, sd, _ = leader1_scale_and_derivs(t)
        v0 = leader0_velocity(t)
        ref_dir = sim.X_ref[1] - sim.X_ref[0]
        ref_dir_norm = np.linalg.norm(ref_dir)
        if ref_dir_norm < 1e-9:
            ref_unit = np.array([1.0, 0.0, 0.0])
        else:
            ref_unit = ref_dir / ref_dir_norm
        return v0 + sd * r01_ref * ref_unit

    def leader1_acceleration(t):
        # a1 = a0 + s_ddot * r01_ref * ref_unit
        _, _, sdd = leader1_scale_and_derivs(t)
        a0 = leader0_acceleration(t)
        ref_dir = sim.X_ref[1] - sim.X_ref[0]
        ref_dir_norm = np.linalg.norm(ref_dir)
        if ref_dir_norm < 1e-9:
            ref_unit = np.array([1.0, 0.0, 0.0])
        else:
            ref_unit = ref_dir / ref_dir_norm
        return a0 + sdd * r01_ref * ref_unit

    # Allocate state arrays
    n_steps = int(T / dt)
    sim.X = np.zeros((n_agents, dim_state, n_steps))
    dX = np.zeros_like(sim.X)      # velocities
    ddX = np.zeros_like(sim.X)     # accelerations

    # set initial states: followers from x0, leaders overwritten with leader positions at t=0
    sim.X[:, :, 0] = x0
    sim.X[0, :, 0] = leader0_position(0.0)
    sim.X[1, :, 0] = leader1_position(0.0)
    dX[0, :, 0] = leader0_velocity(0.0)
    dX[1, :, 0] = leader1_velocity(0.0)
    ddX[0, :, 0] = leader0_acceleration(0.0)
    ddX[1, :, 0] = leader1_acceleration(0.0)

    # --- Simulation Loop ---
    print("Running follower simulation with two leaders (translation + scale)...")
    t0 = time.time()
    leader_indices = (0, 1)

    for k in range(n_steps - 1):
        t = k * dt

        # 1) Set both leaders' states at time t
        sim.X[0, :, k] = leader0_position(t)
        dX[0, :, k] = leader0_velocity(t)
        ddX[0, :, k] = leader0_acceleration(t)

        sim.X[1, :, k] = leader1_position(t)
        dX[1, :, k] = leader1_velocity(t)
        ddX[1, :, k] = leader1_acceleration(t)

        # 2) Compute accelerations (commands) for all agents using the consensus law
        #    (followers will use the leaders' ddX as feedforward terms)
        u = sim.consensus_leader_vel_varying_law(sim.X[:, :, k], dX[:, :, k], ddX[:, :, k], kp=kp, kv=kv)

        # 3) Integrate followers only (agents 2..n_agents-1). Leaders follow their own trajectories.
        for i in range(2, n_agents):
            # treat u[i] as acceleration command: ddX
            ddX[i, :, k] = u[i]

            # semi-implicit Euler integration (stable for second-order systems)
            dX[i, :, k+1] = dX[i, :, k] + dt * ddX[i, :, k]
            sim.X[i, :, k+1] = sim.X[i, :, k] + dt * dX[i, :, k+1]

        # 4) propagate leaders to next time step using their analytic trajectories
        sim.X[0, :, k+1] = leader0_position((k+1)*dt)
        dX[0, :, k+1] = leader0_velocity((k+1)*dt)
        ddX[0, :, k+1] = leader0_acceleration((k+1)*dt)

        sim.X[1, :, k+1] = leader1_position((k+1)*dt)
        dX[1, :, k+1] = leader1_velocity((k+1)*dt)
        ddX[1, :, k+1] = leader1_acceleration((k+1)*dt)

        # Optional: simple safety / debug checks (uncomment if you want to monitor)
        # if np.any(~np.isfinite(sim.X[:, :, k+1])):
        #     print("Non-finite state detected at step", k)
        #     break

    t1 = time.time()
    print(f"Simulation finished in {t1 - t0:.2f} s")

    # Plot 3D trajectories
    # sim.plot_gif_3d(T, dt, max_frames=200)

