"""Generate trajectories from the rope physics engine and create torch geometric datasets

Credit to https://colab.research.google.com/drive/1hirUfPgLU35QCSQSZ7T2lZMyFMOaK_OF?usp=sharing#scrollTo=b2HrUjPnsF_4
and https://github.com/YunzhuLi/CompositionalKoopmanOperators
"""
import os
import json
import numpy as np

from .base import *
from simulator.systems import RopeEngine
from simulator.utils import finite_difference


def generate_trajectories(params, mode='train'):
    """Generate rollouts for the rope system and save into a dataset
    """
    n_rollout = params.n_rollout
    time_steps = params.timesteps
    state_dim = params.state_dim
    dt = params.dt
    video = params.video

    engine = RopeEngine(params)

    for i in range(n_rollout):

        # rollout_dir = os.path.join(data_dir, str(rollout_idx))
        # os.system('mkdir -p ' + rollout_dir)

        engine.init()
        trajectory = np.zeros((time_steps,engine.n_ball, state_dim))

        for j in range(time_steps):
            states = engine.get_state()

            pos = states[:, 0:2].copy()
            vec = states[:, 2:4].copy()

            trajectory[j, :, 0:2] = pos
            trajectory[j, :, 2:4] = vec

            engine.step()

        # estimate the acceleration
        trajectory[..., 4:6] = finite_difference(trajectory[..., 2:4], dt)
        trajectory[:, 1:, 4:6] = (0, engine.gravity)

        if video:
            engine.render(trajectory, path="figures/rope")

    # TODO


class RopeOneStepDataset(pyg.data.Dataset):

    def __init__(self, data_path, split, window_length=7, noise_std=0.0, return_pos=False):
        super().__init__()

        # load dataset from the disk
        with open(os.path.join(data_path, "metadata.json")) as f:
            self.metadata = json.load(f)
        with open(os.path.join(data_path, f"{split}_offset.json")) as f:
            self.offset = json.load(f)
        self.offset = {int(k): v for k, v in self.offset.items()}
        self.window_length = window_length
        self.noise_std = noise_std
        self.return_pos = return_pos

        self.particle_type = np.memmap(os.path.join(data_path, f"{split}_particle_type.dat"), dtype=np.int64, mode="r")
        self.position = np.memmap(os.path.join(data_path, f"{split}_position.dat"), dtype=np.float32, mode="r")

        for traj in self.offset.values():
            self.dim = traj["position"]["shape"][2]
            break

        # cut particle trajectories according to time slices
        self.windows = []
        for traj in self.offset.values():
            size = traj["position"]["shape"][1]
            length = traj["position"]["shape"][0] - window_length + 1
            for i in range(length):
                desc = {
                    "size": size,
                    "type": traj["particle_type"]["offset"],
                    "pos": traj["position"]["offset"] + i * size * self.dim,
                }
                self.windows.append(desc)

    def len(self):
        return len(self.windows)

    def get(self, idx):
        # load corresponding data for this time slice
        window = self.windows[idx]
        size = window["size"]
        particle_type = self.particle_type[window["type"]: window["type"] + size].copy()
        particle_type = torch.from_numpy(particle_type)
        position_seq = self.position[window["pos"]: window["pos"] + self.window_length * size * self.dim].copy()
        position_seq.resize(self.window_length, size, self.dim)
        position_seq = position_seq.transpose(1, 0, 2)
        target_position = position_seq[:, -1]
        position_seq = position_seq[:, :-1]
        target_position = torch.from_numpy(target_position)
        position_seq = torch.from_numpy(position_seq)
        
        # construct the graph
        with torch.no_grad():
            graph = preprocess(particle_type, position_seq, target_position, self.metadata, self.noise_std)
        if self.return_pos:
          return graph, position_seq[:, -1]
        return graph
    

class RopeRolloutDataset(pyg.data.Dataset):

    def __init__(self, data_path, split, window_length=7):
        super().__init__()
        
        # load data from the disk
        with open(os.path.join(data_path, "metadata.json")) as f:
            self.metadata = json.load(f)
        with open(os.path.join(data_path, f"{split}_offset.json")) as f:
            self.offset = json.load(f)
        self.offset = {int(k): v for k, v in self.offset.items()}
        self.window_length = window_length

        self.particle_type = np.memmap(os.path.join(data_path, f"{split}_particle_type.dat"), dtype=np.int64, mode="r")
        self.position = np.memmap(os.path.join(data_path, f"{split}_position.dat"), dtype=np.float32, mode="r")

        for traj in self.offset.values():
            self.dim = traj["position"]["shape"][2]
            break

    def len(self):
        return len(self.offset)

    def get(self, idx):
        traj = self.offset[idx]
        size = traj["position"]["shape"][1]
        time_step = traj["position"]["shape"][0]
        particle_type = self.particle_type[traj["particle_type"]["offset"]: traj["particle_type"]["offset"] + size].copy()
        particle_type = torch.from_numpy(particle_type)
        position = self.position[traj["position"]["offset"]: traj["position"]["offset"] + time_step * size * self.dim].copy()
        position.resize(traj["position"]["shape"])
        position = torch.from_numpy(position)
        data = {"particle_type": particle_type, "position": position}
        return data