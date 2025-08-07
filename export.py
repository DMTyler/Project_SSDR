import numpy as np
import os

def export_ssdr(export_path: str, P: np.ndarray, W: np.ndarray, R: np.ndarray, T: np.ndarray) -> None:
    """
    Save SSDR results to a .npz file.

    Parameters:
    - npz_path: Path to output .npz file. Directory will be created if needed.
    - P:   (V,3) array of refined rest-pose vertices.
    - W:   (V,B) array of vertex-bone weights.
    - R:   (B,F,3,3) array of per-bone per-frame rotations.
    - T:   (B,F,3)   array of per-bone per-frame translations.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    np.savez_compressed(export_path, P=P, W=W, R=R, T=T)

    print(f"SSDR results exported to {export_path}")
