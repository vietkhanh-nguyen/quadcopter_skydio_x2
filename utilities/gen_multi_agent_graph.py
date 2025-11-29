import numpy as np

def build_universal_rigid_graph(n, d):
    """
    Xây dựng universal rigid graph tối thiểu trong không gian d-chiều.
    
    Input:
        n: số node
        d: số chiều (2 hoặc 3)
    Output:
        num_edges: số cạnh
        A: adjacency matrix (n x n)
    """
    if n < d + 1:
        raise ValueError("Số node phải >= d + 1 để có universal rigid graph")
    
    edges = []
    # --- Bước 1: tạo simplex ban đầu ---
    # Simplex gồm d+1 nodes, fully connected
    for i in range(d+1):
        for j in range(i+1, d+1):
            edges.append((i, j))
    
    num_vertices = d + 1
    # --- Bước 2: thêm node mới bằng Henneberg type I ---
    while num_vertices < n:
        # Chọn ngẫu nhiên d nodes hiện có để nối node mới
        attach_nodes = np.random.choice(num_vertices, size=d, replace=False)
        new_node = num_vertices
        for u in attach_nodes:
            edges.append((new_node, u))
        num_vertices += 1

    # --- Bước 3: tạo adjacency matrix ---
    A = np.zeros((n, n), dtype=int)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1  # đồ thị vô hướng

    return len(edges), A



def gen_sphere(n_agents, dim_state,radius):

    X_ref = np.zeros((n_agents, dim_state))          # sphere radius
    offset = 2.0 / n_agents
    increment = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n_agents):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)

        phi = i * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        # scale to desired radius
        pos = radius * np.array([x, y, z])
        X_ref[i, :] = pos

    return X_ref

def gen_rectangle(n_agents, spacing=2.0, agents_per_row=3):
    X_ref = np.zeros((n_agents, 3))

    for i in range(n_agents):
        x = (i % agents_per_row) * spacing
        y = (i // agents_per_row) * spacing
        X_ref[i, :3] = [x, y, 0]  # z = 0 plane

    return X_ref

def gen_icosahedron(radius):
    """
    Initialize X_ref using a regular Icosahedron (12 vertices).
    Only works if n_agents = 12.
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    # Unscaled vertices of icosahedron
    verts = np.array([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],

        [0, -1,  phi],
        [0,  1,  phi],
        [0, -1, -phi],
        [0,  1, -phi],

        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1],
    ], dtype=float)

    # Normalize to radius = 0.5
    norms = np.linalg.norm(verts, axis=1)
    X_ref = radius * verts / norms[:, np.newaxis]
    X_ref = np.vstack((X_ref, np.array([1.0, 1.0, 1.0]), np.array([[0.0, 0.0, 0.0]])))  

    return X_ref

if __name__ == "__main__":

    edges, A = build_universal_rigid_graph(n=12, d=3)

    print("Edges:")
    print(edges)

    print("\nAdjacency matrix:")
    print(A)
