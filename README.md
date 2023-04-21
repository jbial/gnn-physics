# Modeling complex dynamical systems with graph neural networks (GNNs)

### Setup
```
pip install -e .
pip install -r requirements.txt
```

### Datasets
For the SandRamps dataset, visit https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published//PRJ-3702/SandRamps/dataset to download the SandRamps dataset which consists of the following files:
```
metadata.json
train.npz
test.npz
valid.npz
```
Finally, place the files in `datasets/SandRamps` via: 
```
mkdir -p datasets/SandRamps
mv [WHEREVER YOU DOWNLOADED THE DATA TO]/metadata.json datasets/SandRamps/
mv [WHEREVER YOU DOWNLOADED THE DATA TO]/{train,test,valid}.npz datasets/SandRamps/
```
The other datasets are generated via the physics engine.

### Examples
To run the mass spring pendulum:
`python3 scripts/run.py system=rope system.dt=0.01 system.timesteps=300 system.noise_std=3e-4 system.generate_data=True batch_size=20 epochs=10 lr=1e-5 save_interval=10000 eval_interval=2500 rollout_interval=2500 model.n_mp_layers=1 random_seed=654 tag=mean model.hidden_size=64`
