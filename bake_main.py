from update import update_ssdr
from export import export_ssdr
from initialization import initialize_ssdr
from reader import read_vertex_animation

ORIGINAL_PATH   = './anims/original/vertex_animation.npz'
EXPORT_PATH     = './anims/exports/skinned_animation.npz'

def main():
    V, P, F = read_vertex_animation(ORIGINAL_PATH)
    R, T, W = initialize_ssdr(V, rest_pose=P, bone_num=10)
    P, W, R, T, history = update_ssdr(V, P, W, R, T, max_outer_iters=100)
    export_ssdr(EXPORT_PATH, P, W, R, T)

if __name__ == '__main__':
    main()