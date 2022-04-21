# UnitedNet


### Package Installation
```
conda create -n untest python=3.7
conda activate untest
conda install -c conda-forge ipykernel leidenalg pydantic python-igraph scanpy tabulate tensorboard tensorflow=2.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # GPU Powered
conda install pytorch torchvision torchaudio -c pytorch # CPU Only
```

For Visual Studio Code
```
conda install -c conda-forge ipywidgets
conda install tensorflow-estimator=2.6.0
```

### Tensorboard Support 
`tensorboard --logdir=saved_runs`

### Run Unittest
`python -m unittest test/run_all.py`

### Remove python cache files
`find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf`