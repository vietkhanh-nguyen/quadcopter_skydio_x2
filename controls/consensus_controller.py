import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 
from utilities.gen_multi_agent_graph import build_universal_rigid_graph, gen_icosahedron, gen_sphere


import numpy as np

class MultiAgentConsensus:
    def __init__(self, n_agents=7, K=1, graph_type="complete", dim_state=3):
        self.n_agents = n_agents
        self.K = K/self.n_agents
        self.impact_dis = 0.3
        self.muy = (1 + self.impact_dis**4)/self.impact_dis**4
        # Initialize system matrices
        self.graph_type = graph_type
        self.dim_state = dim_state
        self._init_matrices()
        self._init_states()

    # ----------------------------------------------------------------------
    def _init_matrices(self):
        """Initialize Laplacian and dynamic matrices."""
        if self.graph_type == "complete":
            A_graph = np.ones((self.n_agents, self.n_agents), dtype=int)
            np.fill_diagonal(A_graph, 0)
            self.num_edges = int(self.n_agents*(self.n_agents - 1)/2)
        elif self.graph_type == "universal_rigid":
            num_edges, A_graph = build_universal_rigid_graph(self.n_agents, self.dim_state)
            self.num_edges = num_edges
            print(num_edges)
        else:
            raise ValueError("graph_type must be 'complete' or 'universal_rigid'.")
        # Ma trận Laplacian
        self.A_graph = A_graph

    # ----------------------------------------------------------------------
    def _init_states(self):

        # state now includes (x, y, z)
        if self.n_agents != 12:
            X_ref = gen_sphere(self.n_agents, self.dim_state, radius=1)
        else:
            X_ref = gen_icosahedron(radius=5)
        # X_ref = gen_sphere(self.n_agents, self.dim_state, radius=1)
        diff = X_ref[:, np.newaxis, :] - X_ref[np.newaxis, :, :]
        norm = np.linalg.norm(diff, axis=2, keepdims=True)
        norm = np.where(norm == 0, 1e-9, norm)   # avoid divide-by-zero
        self.dir_ref = diff / norm
        print(self.dir_ref[1, 0, :])
        self.X_ref = X_ref

    def projection_matrix(self, gij):
        gij = gij.reshape(-1,1)
        eye_matrix = np.eye(self.dim_state)
        return eye_matrix - gij@gij.transpose()
    # ----------------------------------------------------------------------
    def consensus_law(self, X):
        diff_matrix = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2, keepdims=True)
        dist_matrix = np.where(dist_matrix == 0, 1e-9, dist_matrix)   # avoid divide-by-zero
        dir_cur = diff_matrix / dist_matrix
        # print(dist_matrix.shape)
        dist_matrix = dist_matrix.reshape(self.n_agents, self.n_agents)

        u = np.zeros((self.n_agents, self.dim_state)) 
        for i in range(self.n_agents):
            for j in range(self.n_agents):

                e = dist_matrix[i, j]**2 - self.impact_dis**2
                if e <= 0 and i != j:
                    dBij = (-4*self.muy*e)/(1 + e**2)**2 * diff_matrix[i, j, :]
                    u[i] += dBij / (1 - self.muy*(e**2 / (1 + e**2)))

                if self.A_graph[i, j] == 0:
                    continue  
                u[i] -= self.projection_matrix(dir_cur[i, j, :])@self.dir_ref[i, j, :]

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

    # ----------------------------------------------------------------------
    def simulate(self, x0, dt, T):
        self.dt = dt
        self.T = T
        self.n_steps = int(T / dt)
        self.X = np.zeros((self.n_agents, self.dim_state, self.n_steps))
        self.X[:, :, 0] = x0

        # Euler integration
        for k in range(self.n_steps - 1):
            delta_p = self.consensus_law(self.X[:, :, k])       # tính lực đồng thuận
            # delta_p[:2] += np.array([1, 0])
            self.X[:, :, k+1] = self.X[:, :, k] + self.dt * delta_p


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

    def plot_gif_3d(self, max_frames=100):
        """Animate and export the consensus evolution in 3D for all agents."""
        from mpl_toolkits.mplot3d import Axes3D

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

        filename = "consensus_evolution_3d.gif"
        writer = animation.PillowWriter(fps=20)
        anim.save(filename, writer=writer)
        print(f"✅ 3D GIF saved as {filename}")

        plt.close(fig)


    def plot_gif(self, max_frames=100):
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

        filename = "consensus_evolution.gif"
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

    # Parameters
    n_agents = 12
    dim_state = 3    # sphere mode
    K = 2
    dt = 0.02        # integration step
    T = 10.0          # total simulation time (seconds)
    make_gif = True  # set True to save GIF (may be slow)

    # Reproducible randomness
    np.random.seed(12)

    # Create simulator
    sim = MultiAgentConsensus(
        n_agents=n_agents,
        K=K,
        graph_type="universal_rigid",
        dim_state=dim_state
    )

    # Build initial condition x0:
    # start agents near their reference positions with a small random perturbation
    # shape -> (n_agents, dim_state)
    perturb_scale = 0.06
    x0 = sim.X_ref + perturb_scale * np.random.randn(sim.n_agents, sim.dim_state)

    # Optionally re-normalize to lie near the sphere surface (if desired)
    # This keeps them around radius 1 but perturbed radially too.
    radii = np.linalg.norm(x0, axis=1, keepdims=True)
    radii = np.where(radii == 0, 1e-9, radii)
    x0 = x0 / radii  # project back to unit sphere
    x0 = x0 + 0.02 * np.random.randn(*x0.shape)  # small off-surface perturb

    print("Starting simulation with:")
    print(f"  n_agents = {sim.n_agents}, dim_state = {sim.dim_state}")
    print(f"  dt = {dt}, T = {T}, steps = {int(T/dt)}")
    print("  initial pos sample (first agent):", x0[0])

    # Run simulation
    t0 = time.time()
    sim.simulate(x0, dt, T)
    t1 = time.time()
    print(f"Simulation finished in {t1 - t0:.2f} s")

    # Plot final trajectories (2D projection onto x-y plane)
    sim.plot()

    # Plot reference on sphere (requires dim_state == 3)
    # try:
    #     sim.plot_Xref()
    # except ValueError:
    #     pass

    # Optionally save an animated GIF of the trajectories (may be slow)
    # if make_gif:
    #     print("Saving GIF (this may take a while)...")
    #     sim.plot_gif_3d(max_frames=200)
    #     print("GIF done.")
