"""Generate trajectories from the rope physics engine and create torch geometric datasets

Credit to https://colab.research.google.com/drive/1hirUfPgLU35QCSQSZ7T2lZMyFMOaK_OF?usp=sharing#scrollTo=b2HrUjPnsF_4
and https://github.com/YunzhuLi/CompositionalKoopmanOperators
"""
import os
import json
import numpy as np

from .base import *
from tqdm import tqdm
from simulator.systems import RopeEngine
from simulator.utils import finite_difference


class RopeOneStepDataset(ManyBodySystem, pyg.data.Dataset):

    def __init__(self, params, split, return_pos=False):
        super().__init__(params)
        self.split = split
        self.return_pos = return_pos

        if params.generate_data:
            self.generate_trajectories(params, split)

        # load dataset from the disk
        enc = f"N-{params.n_rollout}_T-{params.timesteps}_D-{params.dt}"
        self.trajectories = np.load(os.path.join(params.data_dir, f"{split}_trajectories_{enc}.npz")) 
        self.particle_type = [np.asarray([0]+[1]*(self.trajectories[t].shape[1]-1)) for t in self.trajectories]

        # cut particle trajectories according to time slices
        self.windows = []
        for t in self.trajectories:
            length = params.timesteps - self.window_length + 1
            for i in range(length):
                desc = {
                    "index": t,  # trajectory index
                    "range": slice(i, i + self.window_length)  # time slice
                }
                self.windows.append(desc)

        # estimate acceleration statistics TODO: correct this
        self.acc_mean = 0
        self.acc_std = 1

    @staticmethod
    def generate_trajectories(params, split, render=False):
        """Generate rollouts for the rope system and save into a dataset
        """
        time_steps = params.timesteps
        state_dim = params.state_dim

        os.makedirs(params.data_dir, exist_ok=True)

        dataset = dict()
        engine = RopeEngine(params)
        print("Generating pendulum trajectories...")
        for i in tqdm(range(params.n_rollout), desc="Rolling out trajectories..."):

            # generate new system parameters
            engine.init()
            trajectory = np.zeros((time_steps, engine.n_ball, state_dim))

            for j in range(time_steps):
                states = engine.get_state()

                pos = states[:, 0:2].copy()
                vec = states[:, 2:4].copy()

                trajectory[j, :, 0:2] = pos
                trajectory[j, :, 2:4] = vec

                engine.step()

            # estimate the acceleration
            trajectory[..., 4:6] = finite_difference(trajectory[..., 2:4], engine.dt)
            trajectory[:, 1:, 4:6] = (0, engine.gravity)

            if render:
                engine.render(trajectory, path=f"figures/rope_{split}_{i}")

            dataset[str(i)] = trajectory

        # save the generated trajectories and encode some parameters in the path
        # N: number of rollouts
        # T: number of timesteps per rollout
        # D: timestep (dt)
        param_str = f"N-{params.n_rollout}_T-{params.timesteps}_D-{engine.dt}"
        data_path = os.path.join(params.data_dir, f"{split}_trajectories_{param_str}.npz")
        np.savez_compressed(data_path, **dataset)

    def velocity(self, window):
        """Get velocity from trajectory
        """
        velocity_seq = self.trajectories[window["index"]][window["range"], :, 2:4]
        return torch.from_numpy(velocity_seq[:-2].transpose(1, 0, 2))
    
    def acceleration(self, window, noise):
        """Ground truth for training
        """
        acceleration = self.trajectories[window["index"]][window["range"], :, 4:6][-1]
        acceleration = torch.from_numpy(acceleration)
        acceleration += (noise[:, -1] / (2*self.params.dt))  # corrupt with noise
        acceleration = (acceleration - torch.tensor(self.acc_mean)) \
            / torch.sqrt(torch.tensor(self.acc_std)**2 + self.noise_std**2)
        return acceleration

    def generate_noise(self, velocity_seq):
        """Generate noise for a trajectory
        """
        time_steps = velocity_seq.shape[1]
        velocity_noise = torch.randn_like(velocity_seq) * (self.noise_std / time_steps**0.5)
        velocity_noise = velocity_noise.cumsum(dim=1)  # random walk
        initial = velocity_noise[:, 0].unsqueeze(1)
        position_noise = torch.cat(  # integrate the noise
            [initial, initial + self.params.dt*velocity_noise[:, 1:].cumsum(dim=1)], dim=1)
        position_noise = torch.cat((torch.zeros_like(position_noise)[:, 0:1], position_noise), dim=1)
        return position_noise

    def adjacency_list(self, size):
        """Adjacency list for link graph (with self loops)
        """
        node_indices = torch.arange(size)
        self_loops = node_indices.expand(2, -1)
        out_edges = torch.stack([node_indices[1:], node_indices[:-1]], dim=0)
        in_edges = out_edges.roll(1, dims=0)
        return torch.cat([self_loops, out_edges, in_edges], dim=1)

    def node_features(self, position, velocity_seq):
        """Node-level features: velocity, distance to the boundary
        """
        return torch.cat((velocity_seq.reshape(velocity_seq.shape[0], -1), position), dim=-1)
    
    def edge_features(self, position, edge_index):
        """Edge-level features: displacement, distance
        """
        dim = position.shape[-1]
        edge_displacement = (torch.gather(position, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1, dim)) -
                    torch.gather(position, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1, dim)))
        edge_displacement /= self.params.rest_length
        edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)
        return torch.cat((edge_displacement, edge_distance), dim=-1)
    
    def preprocess(self, window, particle_type, position_seq, target_position):
        """Preprocess a trajectory and construct the graph (OVERWRITES THE BASE)
        """
        velocity_seq = self.velocity(window)
        position_noise = self.generate_noise(velocity_seq)
        position_seq += position_noise

        # calculate the velocities of particles
        recent_position = position_seq[:, -1]
        
        edge_index = self.adjacency_list(len(particle_type))
        node_features = self.node_features(recent_position, velocity_seq)
        edge_features = self.edge_features(recent_position, edge_index)

        if target_position is not None:
            acceleration = self.acceleration(window, position_noise)
        else:
            acceleration = None

        # return the graph with features
        graph = pyg.data.Data(
            x=particle_type,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=acceleration,
            pos=node_features
        )
        return graph

    def len(self):
        return len(self.windows)

    def get(self, idx):
        # load corresponding data for this time slice
        window = self.windows[idx]
        particle_type = self.particle_type[int(window["index"])]
        particle_type = torch.from_numpy(particle_type)
        position_seq = self.trajectories[window["index"]][window["range"], :, 0:2]
        position_seq = position_seq.transpose(1, 0, 2)
        target_position = position_seq[:, -1]
        position_seq = position_seq[:, :-1]
        target_position = torch.from_numpy(target_position)
        position_seq = torch.from_numpy(position_seq)
        
        # construct the graph
        with torch.no_grad():
            graph = self.preprocess(window, particle_type, position_seq, target_position)
        if self.return_pos:
          return graph, position_seq[:, -1]
        return graph
    

class RopeRolloutDataset(pyg.data.Dataset):

    def __init__(self, params, split):
        super().__init__()
        
        # load data from the disk
        with open(os.path.join(params.data_path, "metadata.json")) as f:
            self.metadata = json.load(f)
        with open(os.path.join(params.data_path, f"{split}_offset.json")) as f:
            self.offset = json.load(f)
        self.offset = {int(k): v for k, v in self.offset.items()}

        self.particle_type = np.memmap(
            os.path.join(params.data_path, f"{split}_particle_type.dat"), dtype=np.int64, mode="r")
        self.position = np.memmap(
            os.path.join(params.data_path, f"{split}_position.dat"), dtype=np.float32, mode="r")

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