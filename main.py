from update import update_ssdr
from initialization import initialize_ssdr
from reader import read_vertex_animation
from player import play_skinning
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(script_dir, 'anims', 'vertex_animation.npz')
    V, P, F = read_vertex_animation(npz_path)

    R, T, W = initialize_ssdr(V, rest_pose=P, bone_num=10)
    P, W, R, T, history = update_ssdr(V, P, W, R, T, max_outer_iters=100)
    play_skinning(P, W, R, T, F, fps=30, cam_offset=-2)

if __name__ == '__main__':
    main()