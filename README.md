# UnitedNet


### Package Installation
```
conda create -n untest python=3.7
conda activate untest
conda install -c conda-forge ipykernel leidenalg pydantic python-igraph scanpy tabulate tensorboard
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # GPU Powered
conda install pytorch torchvision torchaudio -c pytorch # CPU Only
conda install tensorflow
```


### Tensorboard Support 
`tensorboard --logdir=saved_runs`