# UnitedNet


### Package Installation
```
conda create -n untest python=3.7
conda activate untest
conda install -c conda-forge leidenalg pydantic python-igraph scanpy tensorboard
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
conda install tensorflow
```


### Tensorboard Support 
`tensorboard --logdir=saved_runs`