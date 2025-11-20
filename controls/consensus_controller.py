import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D 


import numpy as np

class MultiAgentConsensus:
    def __init__(self, n_agents=7, K=1):
        self.n_agents = n_agents
        self.dim_state = 3
        self.K = K
        self.impact_dis = 1.0 # Robot radius = 0.4
        self.muy = (1 + self.impact_dis**4)/self.impact_dis**4
        # Initialize system matrices
        self._init_matrices()
        self._init_states()

    # ----------------------------------------------------------------------
    def _init_matrices(self):
        """Initialize Laplacian and dynamic matrices."""
        A_graph = np.ones((self.n_agents, self.n_agents), dtype=int)  # tất cả là 1
        np.fill_diagonal(A_graph, 0)  # không nối chính nó

        # Ma trận Laplacian
        D = np.diag(np.sum(A_graph, axis=1))  # ma trận bậc
        Laplace = D - A_graph
        self.L = Laplace
        self.n_edges = np.trace(self.L) // 2 

    # ----------------------------------------------------------------------
    def _init_states(self):
        # Reference positions on a circle (x, y for each agent)
        X_ref = np.zeros(self.n_agents * self.dim_state)
        radius = 4
        self.dis_ref = np.zeros((self.n_agents, self.n_agents))

        for i in range(self.n_agents):
            angle_i = 2 * np.pi * i / self.n_agents
            X_ref[i*2:i*2+2] = radius * np.array([np.cos(angle_i), np.sin(angle_i)])
        
        # Tính ma trận khoảng cách tham chiếu
        for i in range(self.n_agents):
            xi = X_ref[i*2 : i*2 + 2]
            for j in range(i+1, self.n_agents):
                xj = X_ref[j*2 : j*2 + 2]
                dist = np.linalg.norm(xi - xj)
                self.dis_ref[i, j] = dist
                self.dis_ref[j, i] = dist

    # ----------------------------------------------------------------------
    def consensus_law(self, X):
        X_reshaped = X.reshape(self.n_agents, self.dim_state)

        # dist[i,j] = ||X[i] - X[j]||
        diff = X_reshaped[:, np.newaxis, :] - X_reshaped[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        force_factor = np.zeros_like(dist_matrix)
        force_factor = dist_matrix - self.dis_ref
        u = np.zeros((self.n_agents, self.dim_state)) 
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue  
                u[i] -= force_factor[i, j] * diff[i, j, :]
                e = dist_matrix[i, j]**2 - self.impact_dis**2
                if e <= 0:
                    dBij = (-4*self.muy*e)/(1 + e**2)**2 * diff[i, j, :]
                    u[i] += dBij / (1 - self.muy*(e**2 / (1 + e**2)))
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
        """Plot the consensus evolution in the x1–x2 plane for all agents."""
        t = np.arange(self.n_steps) * self.dt
        fig, ax = plt.subplots(figsize=(8, 6))

        # for i in range(self.n_agents):
        #     ax.plot(self.X[i, 0, :], self.X[i, 1, :], label=f"Agent {i + 1}")

        for i in range(self.n_agents):
            ax.scatter(self.X[i, 0, -1], self.X[i, 1, -1])

        ax.set_xlabel("x1 (position)")
        ax.set_ylabel("x2 (position)")
        ax.set_title("Consensus evolution of all agents in state space")
        ax.grid(True)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

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







# ----------------------------------------------------------------------
# Example test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    n_agents = 10
    np.random.seed(1210)
    sim = MultiAgentConsensus(
        n_agents=n_agents,
        K = .5
    )
    dt=1e-2
    T=100.0       
    x0 = 10*np.random.rand(n_agents, 2)
    # x0 = np.array([
    #     [1, 0],
    #     [2, 0],
    #     [4, 0],
    #     [6, 0]
    # ])
    sim.simulate(x0, dt, T)  # chạy mô phỏng
    sim.plot()      # vẽ quỹ đạo
    # sim.plot_gif(max_frames=100)  # tạo gif
