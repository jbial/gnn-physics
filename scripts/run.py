import torch
import hydra
import torch_geometric as pyg

from simulator.utils import set_seed
from simulator.data import RopeOneStepDataset, WaterDropOneStepDataset
from simulator.model import LearnedSimulator


@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(hparams):

    set_seed(hparams.random_seed)

    print(hparams)

    train_dataset = RopeOneStepDataset(hparams.system, 'train')
    hparams.system.n_rollout = 5
    valid_dataset = RopeOneStepDataset(hparams.system, 'valid')
    hparams.system.n_rollout = 10
    test_dataset = RopeOneStepDataset(hparams.system, 'test')

    print(train_dataset.get(0))
    print(test_dataset.get(0))
    print(valid_dataset.get(0))





if __name__ == '__main__':
    main()