"""Contains train function
"""
import os
import torch

from tqdm import tqdm
from simulator.evaluation import rolloutMSE, oneStepMSE


def train(params, simulator, train_loader, valid_loader, valid_rollout_dataset):
    model_path = f"models/{params.system.data_dir.split('/')[-1]}"
    os.makedirs(model_path, exist_ok=True)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1/5e6))

    # recording loss curve
    train_loss_list = []
    eval_loss_list = []
    onestep_mse_list = []
    rollout_mse_list = []
    total_step = 0
    best_rollout = float('inf')

    for i in range(params.epochs):
        simulator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {i}")
        total_loss = 0
        batch_count = 0
        for data in progress_bar:
            optimizer.zero_grad()
            # data = data.cuda()
            pred = simulator(data)
            kinematic_mask = (data.x != 0)
            squared_diff = (pred-data.y).square()
            loss = squared_diff[kinematic_mask].sum() / kinematic_mask.sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / batch_count, "lr": optimizer.param_groups[0]["lr"]})
            total_step += 1
            train_loss_list.append((total_step, loss.item()))

            # evaluation
            if total_step % params.eval_interval == 0:
                simulator.eval()
                eval_loss, onestep_mse = oneStepMSE(
                    simulator, valid_loader, valid_rollout_dataset.acc_std, valid_rollout_dataset.noise_std)
                eval_loss_list.append((total_step, eval_loss))
                onestep_mse_list.append((total_step, onestep_mse))
                tqdm.write(f"Eval: Loss: {eval_loss}, One Step MSE: {onestep_mse}")
                simulator.train()

            # do rollout on valid set
            if total_step % params.rollout_interval == 0:
                simulator.eval()
                rollout_mse = rolloutMSE(simulator, valid_rollout_dataset)
                rollout_mse_list.append((total_step, rollout_mse))
                tqdm.write(f"Eval: Rollout MSE: {rollout_mse}")
                simulator.train()

                if rollout_mse < best_rollout:
                    torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join(model_path, f"{params.tag}_best_val_rollout.pt")
                )
                best_rollout = min(rollout_mse, best_rollout)
                    
                    
            # save model
            if total_step % params.save_interval == 0 or (total_step == len(train_loader)*params.epochs-1):
                torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join(model_path, f"checkpoint_{total_step}.pt")
                )
    return train_loss_list, eval_loss_list, onestep_mse_list, rollout_mse_list