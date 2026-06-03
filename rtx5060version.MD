# incompatibility with gpu rtx 5060 fix

```bash
conda create -n hivt_new python=3.10
conda activate hivt_new
```

```bash
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```
check:
```python
import torch
print(torch.cuda.get_device_capability(0))
```

```bash
pip install torch-geometric
pip install pytorch-lightning==1.9.5
pip install numpy pandas tqdm scipy
```

if broken

```bash 
pip uninstall pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -y
pip install --force-reinstall torch-geometric
```

## argoverse api
1. remove lapsolver dep from setup.py
```bash 
pip install numpy==1.24.4
pip install -e ~/argoverse-api --no-deps
pip install colour descartes h5py hydra-core==1.1.0 \
imageio matplotlib motmetrics==1.1.3 numba
pip install polars pyntcloud scikit-learn shapely
pip install opencv-python==4.7.0.72
```
## downgrade pytorch-lightning, torchmetrics, numpy 
```bash
pip install torchmetrics==0.6.0
pip install pytorch-lightning==1.5.2
pip install numpy==1.24.4 --force-reinstall
```

## code changes



### datamodules/argoverse_v1_datamodule.py

Change:

`from torch_geometric.data import DataLoader`

👉 to:

`from torch_geometric.loader import DataLoader`

### INVALID LR SCHEDULER

in `HiVT/models/hivt.py`
change `return [optimizer], [scheduler]` to:
```python
return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
            }
        }
```

### HiVT/datasets/argoverse_v1_dataset.py
change `return torch.load(self.processed_paths[idx])` to 
`return torch.load(self.processed_paths[idx], weights_only=False)`

**change**
### in HiVT/utils.py
TemporalData.__inc__() 
🔍 Find this method:
`def __inc__(self, key, value):`
✏️ Change it to:
`def __inc__(self, key, value, *args, **kwargs):`


### in HiVT/models/local_encoder.py
🔍 Find:
`def forward(self, ...):`
✏️ Change it to:
`def forward(self, ..., is_causal=False, **kwargs):`
