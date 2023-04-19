"""Base functions for all dataset
"""
import torch
import torch_geometric as pyg

from abc import abstractmethod


class ManyBodySystem:

    def __init__(self, params):
        self.params = params
        self.dim = params.dim
        self.window_length = params.window_length
        self.noise_std = params.noise_std

    @abstractmethod
    def velocity(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def acceleration(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate_noise(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def adjacency_list(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def node_features(self, *args, **kwargs):
        raise NotImplementedError

    def preprocess(self, particle_type, position_seq, target_position):
        """Preprocess a trajectory and construct the graph
        """
        position_noise = self.generate_noise(position_seq)
        position_seq += position_noise

        # calculate the velocities of particles
        recent_position = position_seq[:, -1]
        velocity_seq = self.velocity(position_seq)
        edge_index = self.adjacency_list(recent_position)
        node_features = self.node_features(position_seq, velocity_seq)
        edge_features = self.edge_features(position_seq, velocity_seq, edge_index)

        if target_position is not None:
            acceleration = self.acceleration(recent_position, target_position, velocity_seq, position_noise)
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
