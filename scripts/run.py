import time
import hydra
import numpy as np

from tqdm import tqdm
from simulator.systems.rope import RopeEngine

def gen_Rope(args):
    thread_idx = 0
    n_rollout, time_step = args['n_rollout'], args['time_step']
    dt, video = args['dt'], args['video']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    attr_dim = args.attr_dim  # root, child
    state_dim = args.state_dim  # x, y, xdot, ydot
    action_dim = args.action_dim
    param_dim = args.param_dim  # n_ball, init_x, k, damping, gravity

    act_scale = 2.
    ret_scale = 1.

    # attr, state, action
    engine = RopeEngine(dt, state_dim, action_dim, param_dim)

    for i in range(n_rollout):
        rollout_idx = thread_idx * n_rollout + i

        # num_obj_range = args.num_obj_range if phase in {'train', 'valid'} else args.extra_num_obj_range
        # num_obj = num_obj_range[sub_idx]

        # if rollout_idx % group_size == 0:
        #     engine.init(param=(num_obj, None, None, None, None))
        # else:
        #     while not os.path.isfile(param_file):
        #         time.sleep(0.5)
        #     param = torch.load(param_file)
        #     engine.init(param=param)

        for j in tqdm(range(time_step), desc="Running Simulation..."):
            states_ctl = engine.get_state()[0]
            act_t = np.zeros((engine.num_obj, action_dim))
            act_t[0, 0] = (np.random.rand() * 2 - 1.) * act_scale - states_ctl[0] * ret_scale

            engine.set_action(action=act_t)

            states = engine.get_state()
            actions = engine.get_action()

            n_obj = engine.num_obj

            pos = states[:, :2].copy()
            vec = states[:, 2:].copy()

            '''reset velocity'''
            if j > 0:
                vec = (pos - states_all[j - 1, :, :2]) / dt

            if j == 0:
                attrs_all = np.zeros((time_step, n_obj, attr_dim))
                states_all = np.zeros((time_step, n_obj, state_dim))
                actions_all = np.zeros((time_step, n_obj, action_dim))

            '''attrs: [1, 0] => root; [0, 1] => child'''
            assert attr_dim == 2
            attrs = np.zeros((n_obj, attr_dim))
            # category: the first ball is fixed
            attrs[0, 0] = 1
            attrs[1:, 1] = 1

            assert np.sum(attrs[:, 0]) == 1
            assert np.sum(attrs[:, 1]) == engine.num_obj - 1

            attrs_all[j] = attrs
            states_all[j, :, :2] = pos
            states_all[j, :, 2:] = vec
            actions_all[j] = actions

            engine.step()

        print(states_all.shape)

        engine.render(states_all, path="figures/rope_test")



@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(hparams):

    print(hparams)
    gen_Rope(hparams.systems)


if __name__ == '__main__':
    main()