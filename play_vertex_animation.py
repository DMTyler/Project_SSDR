from reader import read_vertex_animation
from player import play_vertex_animation
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(script_dir, 'anims', 'vertex_animation.npz')
    V, P, F = read_vertex_animation(npz_path)
    play_vertex_animation(V, F, fps=24, cam_offset=-2)


if __name__ == '__main__':
    main()