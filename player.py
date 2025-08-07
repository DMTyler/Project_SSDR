import open3d as o3d
import time
import numpy as np

def play_vertex_animation(V, F=None, fps=24, cam_offset=0.0):
    """
    Play vertex animation using Open3D for real-time 3D visualization, including mesh faces and optional gray wireframe.
    Loops until the user closes the window.

    Parameters:
    - V (np.ndarray): Vertex positions over time, shape (F, N, 3).
    - F (np.ndarray, optional): Face indices, shape (M, 3). Default None.
    - fps (int): Frames per second for playback.
    - cam_offset (float): Offset to move the camera down along its up vector.
    """
    import open3d as o3d
    import time

    frames, n_points, _ = V.shape
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Prepare geometry
    if F is not None:
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(V[0]),
            triangles=o3d.utility.Vector3iVector(F)
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        vis.add_geometry(mesh)
        # Wireframe edges
        edges = set()
        for a, b, c in F:
            edges |= {(min(a,b), max(a,b)), (min(b,c), max(b,c)), (min(c,a), max(c,a))}
        lines = np.array(list(edges), dtype=np.int32)
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(V[0]),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector([[0.2,0.2,0.2]]*len(lines))
        vis.add_geometry(line_set)
        geom_mesh, geom_lines = mesh, line_set
    else:
        pcd = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(V[0])
        )
        vis.add_geometry(pcd)
        geom_mesh, geom_lines = pcd, None

    # Rendering options
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on = False

    # Adjust camera
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    extr = params.extrinsic.copy()
    up = extr[:3,1]
    extr[:3,3] -= up * cam_offset
    params.extrinsic = extr
    ctr.convert_from_pinhole_camera_parameters(params)

    interval = 1.0/fps
    frame_idx = 0
    while vis.poll_events():  # returns False once window is closed
        coords = V[frame_idx]
        if F is not None:
            geom_mesh.vertices = o3d.utility.Vector3dVector(coords)
            geom_lines.points = o3d.utility.Vector3dVector(coords)
        else:
            geom_mesh.points = o3d.utility.Vector3dVector(coords)
        vis.update_geometry(geom_mesh)
        if geom_lines:
            vis.update_geometry(geom_lines)
        vis.update_renderer()

        frame_idx = (frame_idx + 1) % frames
        time.sleep(interval)

    vis.destroy_window()


def compute_skinning(P, W, R, T):
    """
    Compute skinned vertex positions V_hat and joint positions J for each frame.
    Parameters:
        R (B, F, 3, 3)
        T (B, F, 3)

    Returns:
        V_hat: (F, V, 3)
        J    : (F, B, 3)
    """
    B, F, _, _ = R.shape
    V = P.shape[0]

    # 1) rest-pose joint positions p_star (weighted centroids)
    W2 = W ** 2  # (V, B)
    denom = W2.sum(axis=0, keepdims=True) + 1e-12  # (1, B)
    p_star = (P[:, None, :] * W2[:, :, None]).sum(axis=0) / denom.T  # (B, 3)

    # 2) skinning vertices
    V_hat = np.zeros((F, V, 3))
    for b in range(B):
        for f in range(F):
            transformed = (R[b, f] @ P.T).T + T[b, f]  # (V,3)
            V_hat[f] += W[:, b:b + 1] * transformed

    # 3) joint positions
    J = np.zeros((F, B, 3))
    for b in range(B):
        for f in range(F):
            J[f, b] = R[b, f] @ p_star[b] + T[b, f]

    return V_hat, J


def play_skinning(P, W, R, T, faces, fps=24, cam_offset=0.0):
    """
    Play skinned mesh + skeleton joint points in Open3D.

    Parameters:
      P  : (V,3) rest-pose vertices
      W  : (V,B) vertex-bone weights
      R  : (B,F,3,3) per-bone per-frame rotations
      T  : (B,F,3)   per-bone per-frame translations
      faces: (M,3) triangle indices
      fps: frames per second
      cam_offset: camera up-vector offset
    """
    V_hat, J = compute_skinning(P, W, R, T)
    F, V, _ = V_hat.shape

    vis = o3d.visualization.Visualizer()
    vis.create_window('SSDR Skinning Animation')

    # Triangle mesh (light gray)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V_hat[0]),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    vis.add_geometry(mesh)

    # Wireframe edges (dark gray)
    edges = set()
    for a, b, c in faces:
        edges |= {(min(a, b), max(a, b)),
                  (min(b, c), max(b, c)),
                  (min(c, a), max(c, a))}
    lines = np.array(list(edges), dtype=np.int32)
    wire = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(V_hat[0]),
        lines=o3d.utility.Vector2iVector(lines)
    )
    wire.colors = o3d.utility.Vector3dVector([[0.2, 0.2, 0.2]] * len(lines))
    vis.add_geometry(wire)

    # Joint points (red)
    joints_pcd = o3d.geometry.PointCloud()
    joints_pcd.points = o3d.utility.Vector3dVector(J[0])
    joints_pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * J.shape[1])
    vis.add_geometry(joints_pcd)

    # Camera setup
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on = False
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    extr = params.extrinsic.copy()
    up = extr[:3, 1]
    extr[:3, 3] -= up * cam_offset
    params.extrinsic = extr
    ctr.convert_from_pinhole_camera_parameters(params)

    interval = 1.0 / fps
    frame = 0
    while vis.poll_events():
        # Update mesh and wireframe
        pts = V_hat[frame]
        mesh.vertices = o3d.utility.Vector3dVector(pts)
        wire.points = o3d.utility.Vector3dVector(pts)
        # Update joint points
        joints_pcd.points = o3d.utility.Vector3dVector(J[frame])

        vis.update_geometry(mesh)
        vis.update_geometry(wire)
        vis.update_geometry(joints_pcd)
        vis.update_renderer()

        frame = (frame + 1) % F
        time.sleep(interval)

    vis.destroy_window()