"""Generate trajectories from the rope physics engine and create torch geometric datasets

Credit to https://colab.research.google.com/drive/1hirUfPgLU35QCSQSZ7T2lZMyFMOaK_OF?usp=sharing#scrollTo=b2HrUjPnsF_4
and https://github.com/YunzhuLi/CompositionalKoopmanOperators
"""
import os
import numpy as np

from .base import *
from tqdm import tqdm
from simulator.systems import RopeEngine
from simulator.utils import finite_difference


class RopeOneStepDataset(pyg.data.Dataset):

    def __init__(self, params, split, acc_mean=[0., 0.], acc_std=[1., 1.], return_pos=False):
        super().__init__()
        self.params = params
        self.dim = params.dim
        self.window_length = params.window_length
        self.noise_std = params.noise_std
        self.split = split
        self.return_pos = return_pos

        self.engine = RopeEngine(params)

        if params.generate_data:
            self.generate_trajectories(self.engine, params, split)

        # load dataset from the disk
        n_rollouts = getattr(params, f"num_{split}_rollouts")
        enc = f"N-{n_rollouts}_T-{params.timesteps}_D-{params.dt}"
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
        if split == 'train':
            self.acc_mean = np.mean([self.trajectories[t][..., 4:6].mean((0, 1)) for t in self.trajectories], 0)
            self.acc_std = np.std([self.trajectories[t][..., 4:6].std((0, 1)) for t in self.trajectories], 0)
        else:
            self.acc_mean = acc_mean
            self.acc_std = acc_std

        self.engines = []

    @staticmethod
    def generate_trajectories(engine, params, split, render=False):
        """Generate rollouts for the rope system and save into a dataset
        """
        time_steps = params.timesteps
        state_dim = params.state_dim

        os.makedirs(params.data_dir, exist_ok=True)

        dataset = dict()
        n_rollouts = getattr(params, f"num_{split}_rollouts")
        for i in tqdm(range(n_rollouts), desc=f"Rolling out {split} trajectories..."):

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
            trajectory[:, 0, 0:2] = trajectory[0, 0, 0:2]  # fix position of first body
            trajectory[:, 0, 2:4] = 0  # fix velocity of first body
            trajectory[..., 4:6] = finite_difference(trajectory[..., 2:4], engine.dt)
            trajectory[0, 1:, 4:6] = (0, engine.gravity)

            if render:
                engine.render(trajectory, path=f"figures/rope_{split}_{i}")

            dataset[str(i)] = trajectory

        # save the generated trajectories and encode some parameters in the path
        # N: number of rollouts
        # T: number of timesteps per rollout
        # D: timestep (dt)
        param_str = f"N-{n_rollouts}_T-{params.timesteps}_D-{engine.dt}"
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
        acceleration += noise[:, -1]  # corrupt with noise
        acceleration[1:] -= torch.tensor(self.acc_mean) \
            / torch.sqrt(torch.tensor(self.acc_std)**2 + self.noise_std**2)
        return acceleration

    def generate_noise(self, velocity_seq, noise_scale):
        """Generate noise for a trajectory
        """
        time_steps = velocity_seq.shape[1]
        velocity_noise = torch.randn_like(velocity_seq) * (noise_scale / time_steps**0.5)
        velocity_noise = velocity_noise.cumsum(dim=1)  # random walk
        initial = velocity_noise[:, 0].unsqueeze(1)
        position_noise = torch.cat(  # integrate the noise
            [initial, initial + self.params.dt*velocity_noise[:, 1:].cumsum(dim=1)], dim=1)
        position_noise = torch.cat((torch.zeros_like(position_noise)[:, 0:1], position_noise), dim=1)
        position_noise[0] = 0
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
        delta_x = position[:, -1] - position[:, -2]
        delta_x_size = delta_x.square().sum(dim=-1, keepdims=True)
        rest_length = self.params.rest_length*torch.ones_like(delta_x_size)
        position_feats = torch.cat([delta_x, delta_x_size, rest_length], dim=-1)
        return torch.cat((velocity_seq.reshape(velocity_seq.shape[0], -1), position_feats), dim=-1)
    
    def edge_features(self, position, edge_index):
        """Edge-level features: displacement, distance
        """
        dim = position.shape[-1]
        edge_displacement = (torch.gather(position, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1, dim)) -
                    torch.gather(position, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1, dim)))
        edge_displacement /= self.params.rest_length
        edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)
        return torch.cat((edge_displacement, edge_distance), dim=-1)
    
    def preprocess(self, window, particle_type, position_seq, target_position, noise_scale):
        """Preprocess a trajectory and construct the graph (OVERWRITES THE BASE)
        """
        velocity_seq = self.velocity(window)
        position_noise = self.generate_noise(velocity_seq, noise_scale)
        position_seq += position_noise

        # calculate the velocities of particles
        recent_position = position_seq[:, -1]
        
        edge_index = self.adjacency_list(len(particle_type))
        node_features = self.node_features(position_seq, velocity_seq).float()
        edge_features = self.edge_features(recent_position, edge_index).float()

        if target_position is not None:
            acceleration = self.acceleration(window, position_noise).float()
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
            graph = self.preprocess(window, particle_type, position_seq, target_position, self.noise_std)
        if self.return_pos:
          return graph, position_seq[:, -1]
        return graph
    

class RopeRolloutDataset(RopeOneStepDataset):

    def __init__(self, params, split, acc_mean=[0.,0.], acc_std=[1.,1.]):
        params.generate_data = (split == 'test')  # only generate for test rollout dataset
        super().__init__(params, split, acc_mean, acc_std)
        
    def len(self):
        return len(self.trajectories)

    def get(self, idx):
        traj = self.trajectories[str(idx)]
        particle_type = self.particle_type[idx].copy()
        particle_type = torch.from_numpy(particle_type)
        position = traj[..., 0:2].copy()
        position = torch.from_numpy(position)
        data = {
            "particle_type": particle_type, 
            "position": position, 
            "traj_index": str(idx)
        }
        return data
    
    def rollout(self, model, data):
        device = next(model.parameters()).device
        model.eval()
        window_size = model.window_size + 1
        total_time = data["position"].shape[0]
        traj = data["position"][:window_size]
        traj = traj.permute(1, 0, 2)
        particle_type = data["particle_type"]

        for t in range(total_time - window_size):
            with torch.no_grad():
                window = {"index": data["traj_index"], "range": slice(t, window_size + 1 + t)}
                graph = self.preprocess(
                    window, particle_type, traj[:, -window_size:], None, 0.0)
                graph = graph.to(device)
                kinematic_mask = (particle_type != 0)
                acceleration = model(graph).cpu()
                acceleration *= torch.sqrt(torch.tensor(self.acc_std)**2 + self.noise_std**2)
                acceleration += torch.tensor(self.acc_mean)
                acceleration *= kinematic_mask.unsqueeze(-1)  # mask out static objects

                # Verlet integration
                recent_position = traj[:, -1]
                recent_velocity = (recent_position - traj[:, -2]) / self.params.dt
                new_velocity = recent_velocity + acceleration*self.params.dt
                new_position = recent_position + new_velocity*self.params.dt
                traj = torch.cat((traj, new_position.unsqueeze(1)), dim=1)

        return traj