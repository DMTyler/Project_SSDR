import numpy as np
from scipy.optimize import minimize

def kabsch(P: np.ndarray, Q: np.ndarray, weights: np.ndarray = None):
    """
    Compute the optimal rotation R and translation T that aligns P -> Q
    using the (optionally weighted) Kabsch algorithm.

    P, Q: arrays of shape (V, 3)
    weights: optional array of shape (V,) for weighted procrustes

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


def bone_transformations_kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray):
    """Return (R, t) that minimises Σ w^2‖Rp+T−q‖².

    Parameters
    ----------
    P : (N,3)  rest‑pose vertices p_i
    Q : (N,3)  target/residual q_i^t  ( **un‑weighted** )
    w : (N,)   per‑vertex weights w_{ij}
    """
    # ---------- centralize (Eq.8(a) and 8(b)) ----------
    w2_sum = np.sum(w ** 2)
    p_star = (w[:, None] ** 2 * P).sum(axis=0) / w2_sum         # Σ w² p / Σ w²
    q_star = (w * Q.T).sum(axis=1) / w2_sum                     # Σ w q / Σ w²
    P_tilde = w[:, None] * (P - p_star)                         # w (p − p*)
    Q_tilde = Q - (w[:, None] * q_star)                         # q − w q*
    # ---------- weighted covariance (Eq.(9)) ----------
    H = P_tilde.T @ Q_tilde                                     # Σ w (p−p*)(q−wq*)ᵀ
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    t = q_star - R @ p_star
    return R, t


def solve_vertex_weights(A, b, x0, tol=1e-9, maxiter=100):
    """
    Solve for a single vertex weight vector x (length B):
      minimize ||A x - b||^2
      subject to x >= 0, sum(x) = 1
    using SLSQP.
    """
    B = A.shape[1]

    # Objective and gradient
    def obj(x):
        r = A @ x - b
        return r.dot(r)
    def grad(x):
        return 2 * A.T @ (A @ x - b)

    # Constraints and bounds
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, None)] * B

    res = minimize(obj, x0, jac=grad,
                   bounds=bounds, constraints=cons,
                   method='SLSQP',
                   options={'ftol': tol, 'maxiter': maxiter})
    if not res.success:
        raise RuntimeError("ASM solver failed: " + res.message)
    return res.x


def update_bone_vertex_weights(verts: np.ndarray, rest: np.ndarray,
                               R: np.ndarray, T: np.ndarray, prev_W: np.ndarray,
                               K: int = 4):
    """
    Update bone-vertex weight map W (V x B), with at most K non-zeros per row.

    Parameters:
        verts        : (F, V, 3) observed positions over frames
        rest         : (V, 3) array of rest-pose vertex positions
        R            : (B, F, 3, 3) per-bone per-frame rotation matrices
        T            : (B, F, 3) per-bone per-frame translation vectors
        prev_W       : (V, B) previous weights for initialization
        K            : max number of bones per vertex

    Returns:
        W_sparse     : (V, B) updated, normalized, K-sparse weight matrix
    """
    V = rest.shape[0]
    B, T_frames, _, _ = R.shape
    W_sparse = np.zeros((V, B))

    # Precompute A^T A and A^T b LU (not shown) for acceleration if desired

    for i in range(V):
        # Build A_i (3T, B) and b_i (3T, 1)
        Ai = np.zeros((3 * T_frames, B))
        bi = verts[:, i, :].reshape(-1)  # (T, V, 3) -> stack to (3T,)

        for j in range(B):
            # Predict rest_pos[i] transformed by bone j over all frames
            # result shape (T, 3)
            transformed = (R[j] @ rest[i] + T[j])
            Ai[:, j] = transformed.reshape(-1)

        x0_full = prev_W[i].copy()

        # Step 1: solve full constrained least squares
        x_full = solve_vertex_weights(Ai, bi, x0_full)

        # Step 2: prune to top-K bones
        topK = np.argsort(x_full)[-K:]
        Ai_K = Ai[:, topK]
        x0_K = x_full[topK]

        # Step 3: re-solve on reduced system
        x_K = solve_vertex_weights(Ai_K, bi, x0_K)

        # Store sparse result
        W_sparse[i, topK] = x_K

    return W_sparse


def reinitialize_bone(
    rest: np.ndarray,
    verts: np.ndarray,
    W: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    j: int,
    reinit_count: dict[int,int],
    max_reinit: int = 10,
    neighbor_k: int = 20,
    eps: float = 3.0
) -> None:
    """
    Reinitialize transforms for a single bone index j within update_bone_transformations.

    Parameters:
        verts: (F, V, 3) array of sampled vertex positions over F frames.
        rest : (V, 3) array of rest-pose vertex positions.
        W    : (V, B) array of vertex-bone weights.
        R    : (B, F, 3, 3) array of current bone rotations (in-place update).
        T    : (B, F, 3) array of current bone translations (in-place update).
        j    : bone index to reinitialize.
        reinit_count: dict mapping bone index to its current reinit count.
        max_reinit : maximum allowed reinitializations per bone.
        neighbor_k : number of nearest-neighbor vertices to use.
        eps        : weight-energy threshold (sum w^2) for invalid bones.

    Behavior:
        - If sum(W[:,j]^2) < eps and reinit_count[j] < max_reinit:
          * Compute per-vertex RMS error over all frames.
          * Select the vertex i0 with highest error.
          * Find neighbor_k nearest neighbors of i0 in rest-pose.
          * For each frame f, perform an unweighted Kabsch on these neighbors:
            R[j,f], T[j,f] = kabsch(P_neighbors, V_neighbors[f], None)
          * Increment reinit_count[j].
        - All updates are done in-place on R and T.
    """
    # Check invalid bone and count
    wj = W[:, j]
    if np.sum(wj**2) >= eps or reinit_count.get(j, 0) >= max_reinit:
        return

    F, V, _ = verts.shape
    B = W.shape[1]

    # Compute per-vertex RMS error over frames
    preds = np.zeros((F, V, 3))
    for f in range(F):
        for b in range(B):
            preds[f] += W[:, b:(b+1)] * (R[b, f] @ rest.T).T + T[b, f]
    errs = np.sqrt(np.mean((verts - preds) ** 2, axis=(0, 2)))  # (V,)

    # Worst-fitting vertex
    i0 = int(np.argmax(errs))

    # K nearest neighbors in rest-pose
    diffs = rest - rest[i0]
    d2 = np.sum(diffs * diffs, axis=1)
    nbrs = np.argsort(d2)[:neighbor_k]

    # Neighbor subsets
    P_nb = rest[nbrs]         # (K,3)
    V_nb_all = verts[:, nbrs]  # (F,K,3)

    # Reinitialize R[j], T[j] across frames
    for f in range(F):
        R_jf, T_jf = kabsch(P_nb, V_nb_all[f], None)
        R[j, f] = R_jf
        T[j, f] = T_jf

    # Increment reinit counter
    reinit_count[j] = reinit_count.get(j, 0) + 1


def update_bone_transformations(verts: np.ndarray,
                                rest: np.ndarray,
                                W: np.ndarray,
                                R_init: np.ndarray | None = None,
                                T_init: np.ndarray | None = None,
                                eps: float = 1e-8,
                                passes: int = 1):
    """
    Block‑coordinate descent updating {R_j^t, T_j^t}.
    rest  : (V,3)
    verts : (F,V,3)
    W     : (V,B)
    returns R (B,F,3,3), T (B,F,3)
    """
    T_frames, V, _ = verts.shape
    B = W.shape[1]

    # initial R/T
    R = np.repeat(np.eye(3)[None, None], B, axis=0) if R_init is None else R_init.copy()
    R = np.repeat(R, T_frames, axis=1)              if R_init is None else R
    T = np.zeros((B, T_frames, 3))                  if T_init is None else T_init.copy()

    P = rest  # alias for clarity

    reinit_count = {} # Used to count how many times an insignificant bone has been reinitialized

    for _ in range(passes):
        # transformed rest per bone‑frame‑vertex
        pr = np.einsum('btij,vj->btvi', R, P)  # (B,T,V,3)
        pr += T[:, :, None, :]
        for t in range(T_frames):
            V_t = verts[t]
            pred_all = np.einsum('vb,bvc->vc', W, pr[:, t]) # (V, 3)
            for j in range(B):
                w_j = W[:, j]
                if np.sum(w_j ** 2) < eps:
                    reinitialize_bone(rest, verts, W, R, T, j, reinit_count, eps=eps)
                    continue  # insignificant bone
                # residual q_i^t
                pred_j = w_j[:, None] * pr[j, t]
                q = V_t - (pred_all - pred_j)
                R_j, T_j = bone_transformations_kabsch(P, q, w_j)
                R[j, t] = R_j
                T[j, t] = T_j
                pr[j, t] = (R_j @ P.T).T + T_j
                pred_all = np.einsum('vb,bvc->vc', W, pr[:, t])

    return R, T


def compute_rms_error(V: np.ndarray,
                      P: np.ndarray,
                      W: np.ndarray,
                      R: np.ndarray,
                      T: np.ndarray) -> float:
    """
    Frame-averaged RMS error, accepting:
      V: (F, N, 3) sampled vertex trajectories
      P: (N, 3)    rest-pose vertices
      W: (N, B)    vertex-to-bone weight matrix
      R: (B, F, 3, 3) per-bone per-frame rotations
      T: (B, F, 3)   per-bone per-frame translations
    Returns
      RMS reconstruction error (scalar).
    """
    F, N, _ = V.shape
    B = W.shape[1]

    # Reconstruct each frame with linear blend skinning
    V_hat = np.zeros_like(V)
    for f in range(F):
        for b in range(B):
            # apply bone b's transform at frame f to rest-pose vertices
            transformed = (R[b, f] @ P.T).T + T[b, f]  # (N,3)
            # accumulate weighted contribution
            V_hat[f] += W[:, b, None] * transformed

    # compute RMS over all frames and vertices
    return 1000 * np.sqrt(np.mean((V - V_hat) ** 2))


def refine_rest_pose(V, W, R, T, P0):
    """
    最后在收敛后，对静态姿态 P 做线性最小二乘修正。

    V: (F, N, 3)
    W: (N, B)
    R: (F, B, 3, 3)
    T: (F, B, 3)
    P0: (N, 3) 初始 rest-pose

    Returns: P: (N, 3) 修正后的 rest-pose
    """
    import numpy as np

    F, N, _ = V.shape
    P = P0.copy()

    for i in range(N):
        A_i = np.zeros((3, 3))
        b_i = np.zeros(3)
        for f in range(F):
            # A_i ← Σ_b w[i,b] * R[b,f]
            A_i += np.einsum('b,bij->ij', W[i], R[:, f, :, :])

            # T_sum ← Σ_b w[i,b] * T[b,f]
            T_sum = np.einsum('b,bi->i', W[i], T[:, f, :])

            # b_i ← b_i + (V[f,i] - T_sum)
            b_i += V[f, i] - T_sum

            # 解 A_i P_i = b_i
        P[i] = np.linalg.lstsq(A_i, b_i, rcond=None)[0]

    return P


def update_ssdr(
        V,
        P,
        W,
        R,
        T,
        *,
        max_outer_iters: int = 100,
        tol: float = 1e-5,
        update_bone_transformation_passes: int = 1,
        verbose: bool = True):
    """
    Alternating optimization driver for SSDR.

    Parameters:
        V : (F,V,3)     采样帧中的已知顶点坐标。
        P : (V,3)       初始静态姿态（会被就地更新）。
        W : (V,B)       顶点-骨骼权重（会被就地更新）。
        R : (F,B,3,3)   每帧每骨骼旋转（会被就地更新）。
        T : (F,B,3)     每帧每骨骼平移（会被就地更新）。
        max_outer_iters 最多进行多少次 (W ↔ R,T) 交替。
        tol             相对 RMS 改进阈值，小于此值即视为收敛。
        update_bone_transformation_passes 每轮更新 R,T 时内部 micro-passes 次数。
        verbose         是否打印迭代日志。

    Returns:
        P,W,R,T,history : tuple
            收敛后的参数 & 每轮 RMS 误差记录。
    """
    history = []
    prev_err = np.inf

    for outer in range(max_outer_iters):
        # --- (1) 固定 (R,T) → 解 W ---
        W = update_bone_vertex_weights(V, P, R, T, prev_W=W)

        # --- (2) 固定 W → 解 (R,T) ---
        R, T = update_bone_transformations(
            V,
            P,
            W,
            R_init=R,
            T_init=T,
            passes=update_bone_transformation_passes,
        )

        # --- (3) 监控误差 ---
        err = compute_rms_error(V, P, W, R, T)
        history.append(err)
        if verbose:
            print(f"[{outer:02d}]  RMS = {err:.6f}")

        # 提前终止判据
        if outer > 0 and np.abs(prev_err - err) / (prev_err + 1e-12) < tol:
            if verbose:
                print("Converged.")
            break

        prev_err = err

    # --- (4) 收敛后再优化一次静态姿态 P ---
    P = refine_rest_pose(V, W, R, T, P)
    if verbose:
        err = compute_rms_error(V, P, W, R, T)
        print(f"[Refined]  RMS = {err:.6f}")

    return P, W, R, T, history