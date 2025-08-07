import threading

from reader import read_vertex_animation, read_skinned_animation
from player import play_vertex_animation, play_skinned_animation

ORIGINAL_PATH   = "./anims/original/vertex_animation.npz"
SKINNED_PATH    = "./anims/exports/skinned_animation.npz"

def play_original():
    V, P, F = read_vertex_animation(ORIGINAL_PATH)
    play_vertex_animation(V, F, fps=24, cam_offset=-2)

def play_skinned():
    _, _, F = read_vertex_animation(ORIGINAL_PATH)
    P, W, R, T = read_skinned_animation(SKINNED_PATH)
    play_skinned_animation(P, W, R, T, faces=F, fps=24, cam_offset=-2)


if __name__ == '__main__':
    threading.Thread(target=play_original).start()
    threading.Thread(target=play_skinned).start()