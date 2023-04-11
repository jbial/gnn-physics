# Modeling complex dynamical systems with graph neural networks (GNNs)

### Setup
First run 
```
pip install -e .
pip install -r requirements.txt
```

### Datasets
Visit https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published//PRJ-3702/SandRamps/dataset to download the SandRamps dataset which consists of the following files:
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
