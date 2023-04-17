"""Creates torch geometric datasets for the WaterDropSmall dataset

Credit to https://colab.research.google.com/drive/1hirUfPgLU35QCSQSZ7T2lZMyFMOaK_OF?usp=sharing#scrollTo=b2HrUjPnsF_4
"""
import os
import json
import numpy as np

from .base import *


class WaterDropOneStepDataset(ManyBodySystem, pyg.data.Dataset):

    def __init__(self, params, split, return_pos=False):
        super().__init__(params)

        # load dataset from the disk
        with open(os.path.join(params.data_dir, "metadata.json")) as f:
            self.metadata = json.load(f)
        with open(os.path.join(params.data_dir, f"{split}_offset.json")) as f:
            self.offset = json.load(f)
        self.offset = {int(k): v for k, v in self.offset.items()}
        
        self.return_pos = return_pos
        self.boundary = torch.tensor(self.metadata["bounds"])

        self.particle_type = np.memmap(
            os.path.join(params.data_dir, f"{split}_particle_type.dat"), dtype=np.int64, mode="r")
        self.position = np.memmap(
            os.path.join(params.data_dir, f"{split}_position.dat"), dtype=np.float32, mode="r")

        # cut particle trajectories according to time slices
        self.windows = []
        for traj in self.offset.values():
            size = traj["position"]["shape"][1]
            length = traj["position"]["shape"][0] - self.window_length + 1
            for i in range(length):
                desc = {
                    "size": size,
                    "type": traj["particle_type"]["offset"],
                    "pos": traj["position"]["offset"] + i * size * self.dim,
                }
                self.windows.append(desc)

    def velocity(self, position_seq):
        """Compute velocity from position
        """
        return position_seq[:, 1:] - position_seq[:, :-1]
    
    def acceleration(self, recent_position, target_position, velocity_seq, noise):
        """Ground truth for training
        """
        last_velocity = velocity_seq[:, -1]
        next_velocity = target_position + noise[:, -1] - recent_position
        acceleration = next_velocity - last_velocity
        acceleration = (acceleration - torch.tensor(self.metadata["acc_mean"])) \
            / torch.sqrt(torch.tensor(self.metadata["acc_std"])**2 + self.noise_std**2)
        return acceleration

    def generate_noise(self, position_seq):
        """Generate noise for a trajectory"""
        velocity_seq = self.velocity(position_seq)
        time_steps = velocity_seq.size(1)
        velocity_noise = torch.randn_like(velocity_seq) * (self.noise_std / time_steps ** 0.5)
        velocity_noise = velocity_noise.cumsum(dim=1)
        position_noise = velocity_noise.cumsum(dim=1)
        position_noise = torch.cat((torch.zeros_like(position_noise)[:, 0:1], position_noise), dim=1)
        return position_noise

    def adjacency_list(self, recent_position):
        """Get nearest neighbor connectivity graph
        """
        n_particle = recent_position.size(0)
        return pyg.nn.radius_graph(
            recent_position, self.metadata["default_connectivity_radius"], loop=True, max_num_neighbors=n_particle)   

    def node_features(self, position_seq, velocity_seq):
        """Node-level features: velocity, distance to the boundary
        """
        recent_position = position_seq[:, -1]
        distance_to_lower_boundary = recent_position - self.boundary[:, 0]
        distance_to_upper_boundary = self.boundary[:, 1] - recent_position
        distance_to_boundary = torch.cat((distance_to_lower_boundary, distance_to_upper_boundary), dim=-1)
        distance_to_boundary = torch.clip(distance_to_boundary / self.metadata["default_connectivity_radius"], -1.0, 1.0)     
        return torch.cat((velocity_seq.reshape(velocity_seq.size(0), -1), distance_to_boundary), dim=-1)
    
    def edge_features(self, position_seq, _, edge_index):
        """Edge-level features: displacement, distance
        """
        recent_position = position_seq[:, -1]
        dim = recent_position.size(-1)
        edge_displacement = (torch.gather(recent_position, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1, dim)) -
                    torch.gather(recent_position, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1, dim)))
        edge_displacement /= self.metadata["default_connectivity_radius"]
        edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)
        return torch.cat((edge_displacement, edge_distance), dim=-1)

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
        target_position = position_seq[:, -1]  # reserve final position as target
        position_seq = position_seq[:, :-1]  # use previous WINDOW_LENGTH-1 as data
        target_position = torch.from_numpy(target_position)
        position_seq = torch.from_numpy(position_seq)

        print(position_seq.shape)
        
        # construct the graph
        with torch.no_grad():
            graph = self.preprocess(particle_type, position_seq, target_position)
        if self.return_pos:
          return graph, position_seq[:, -1]
        return graph
    

class WaterDropRolloutDataset(ManyBodySystem, pyg.data.Dataset):

    def __init__(self, params, split):
        super().__init__(params)
        
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