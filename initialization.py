import numpy as np


def kabsch(P: np.ndarray, Q: np.ndarray, weights: np.ndarray = None):
    """
    Compute the optimal rotation R and translation T that aligns P -> Q
    using the (optionally weighted) Kabsch algorithm.

    P, Q: arrays of shape (N, 3)
    weights: optional array of shape (N,) for weighted procrustes

    Returns:
      R: (3,3) rotation matrix
      t: (3,) translation vector
    """
    assert P.shape == Q.shape, "P and Q must be same shape"
    N = P.shape[0]
    if weights is None:
        w = np.ones(N)
    else:
        w = weights
    # Compute weighted centroids
    w_sum = w.sum()
    p_centroid = (P * w[:,None]).sum(axis=0) / w_sum
    q_centroid = (Q * w[:,None]).sum(axis=0) / w_sum
    # Center the points
    P_centered = P - p_centroid
    Q_centered = Q - q_centroid
    # Weighted covariance matrix
    H = (w[:,None] * P_centered).T @ Q_centered
    # SVD of covariance
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Fix improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = q_centroid - R @ p_centroid
    return R, t


def initialize_ssdr(verts: np.ndarray, *, rest_pose: np.array = None, bone_num: int = 10, num_iters: int = 5, seed: int = 3407):
    """
    SSDR initialization via K-means-like clustering.

    verts: (F, V, 3) array of vertex positions over T frames
    P: (V, 3) rest pose
    B: number of bones (clusters)
    num_iters: number of assignment-update iterations
    seed: random seed for reproducibility, 3407 by default.[1]

    Returns:
      Rs: (B, F, 3, 3) array of per-bone per-frame rotations
      Ts: (B, F, 3)   array of per-bone per-frame translations
      W:  (V, B)       one-hot weight matrix


    References:
      [1] D. Picard, “Torch.manual_seed(3407) is all you need: On the influence of random seeds
          in deep learning architectures for computer vision,” arXiv, Jan. 2021.
          doi:10.48550/arXiv.2109.08203
    """
    np.random.seed(seed)
    T, V, _ = verts.shape
    if rest_pose is None:
        # Choose rest pose as frame 0
        rest = verts[0]
    else:
        rest = rest_pose
    # Random initial assignments.
    assignments = np.random.randint(0, bone_num, size=V)
    # Allocate arrays
    Rs = np.zeros((bone_num, T, 3, 3))
    Ts = np.zeros((bone_num, T, 3))

    for it in range(num_iters):
        # Update step: compute bone transforms
        for j in range(bone_num):
            idx = np.where(assignments == j)[0]
            if idx.size == 0:
                # if no vertices assigned, pick one random to avoid empty cluster
                idx = np.array([np.random.randint(0, V)])
            for t in range(T):
                # Run kabsch algorithm on each frame
                rest_pose = rest[idx]
                Q = verts[t, idx]
                R, tvec = kabsch(rest_pose, Q)
                Rs[j, t] = R
                Ts[j, t] = tvec
        # Assignment step: compute errors and reassign
        errors = np.zeros((V, bone_num))
        for j in range(bone_num):
            for t in range(T):
                # apply current bone transform to all rest vertices
                P_all = rest
                transformed = (Rs[j, t] @ P_all.T).T + Ts[j, t]
                # accumulate squared error
                errors[:, j] += np.sum((verts[t] - transformed)**2, axis=1)
        # reassign each vertex to bone with minimal error
        assignments = np.argmin(errors, axis=1)

    # Build one-hot weight matrix. Each vertex is affected by only one bone, with weight=1
    W = np.zeros((V, bone_num))
    W[np.arange(V), assignments] = 1.0
    return Rs, Ts, W

