import torch
import hydra
import torch_geometric as pyg

from simulator.utils import set_seed
from simulator.data import rope_dataset


@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(hparams):

    set_seed(hparams.random_seed)
    rope_dataset.generate_trajectories(hparams.system)


if __name__ == '__main__':
    main()