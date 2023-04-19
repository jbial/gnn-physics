import time
import torch
import hydra
import torch_geometric as pyg

from simulator.trainer import train
from simulator.utils import set_seed
from simulator.model import LearnedSimulator
from simulator.data import RopeOneStepDataset, RopeRolloutDataset
from simulator.visualization import visualize_losses, visualize_graph


@hydra.main(config_path='../config', config_name='config', version_base=None)
def main(hparams):

    set_seed(hparams.random_seed)
    hparams.system.generate_data = hparams.system.generate_data and (hparams.pretrained_path is None)

    # load dataset
    train_dataset = RopeOneStepDataset(hparams.system, "train")
    valid_dataset = RopeOneStepDataset(hparams.system, "valid", train_dataset.acc_mean, train_dataset.acc_std)
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True, pin_memory=True)
    valid_loader = pyg.loader.DataLoader(valid_dataset, batch_size=hparams.batch_size, shuffle=True, pin_memory=True)
    valid_rollout_dataset = RopeRolloutDataset(hparams.system, "valid", train_dataset.acc_mean, train_dataset.acc_std)
    test_rollout_dataset = RopeRolloutDataset(hparams.system, "test", train_dataset.acc_mean, train_dataset.acc_std)

    # build model
    simulator = LearnedSimulator(hparams.system.dim, hparams.model)
    # simulator = simulator.cuda()

    # train the model
    if hparams.pretrained_path is None:
        start_time = time.time()
        train_loss_list, eval_loss_list, onestep_mse_list, rollout_mse_list = train(
            hparams, simulator, train_loader, valid_loader, valid_rollout_dataset)
        print(f"Finished training in {(time.time()-start_time)/3600:.4f} hours")
        
        # visualize
        visualize_losses(
            [train_loss_list, eval_loss_list],
            ["train", "validation"],
            f"figures/onestep_loss.{hparams.figure.save_format}",
            hparams.figure
        )
        if len(onestep_mse_list) > 0 and len(rollout_mse_list) > 0:
            visualize_losses(
                [onestep_mse_list, rollout_mse_list],
                ["onestep", "rollout"],
                f"figures/MSE.{hparams.figure.save_format}",
                hparams.figure
            )
    else:
        checkpoint = torch.load(hparams.pretrained_path)
        simulator.load_state_dict(checkpoint["model"])

    # rollout on the first trajectory in the rollout dataset
    rollout_data = test_rollout_dataset.get(0)
    rollout_traj = test_rollout_dataset.rollout(simulator, rollout_data)
    rollout_traj = rollout_traj.permute(1, 0, 2)
    test_rollout_dataset.engine.render(rollout_traj, states_gt=rollout_data["position"], path=f"figures/rope_final")


if __name__ == '__main__':
    main()