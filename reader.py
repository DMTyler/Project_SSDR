import numpy as np

def read_vertex_animation(npz_path):
    """
    Load vertex animation data from an NPZ file.

    Parameters:
    - npz_path (str): Path to the .npz file containing arrays 'V' and 'P'.

    Returns:
    - V (np.ndarray): Vertex positions over time, shape (F, N, 3).
    - P (np.ndarray): Initial vertex positions, shape (N, 3).
    - F (np.ndarray): Final vertex positions, shape (M, 3).
    """
    data = np.load(npz_path)
    V = data['V']  # shape: (F, N, 3)
    P = data['P']  # shape: (N, 3)
    F = data['F']
    return V, P, F