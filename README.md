# HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction
This repository contains the official implementation of [HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_HiVT_Hierarchical_Vector_Transformer_for_Multi-Agent_Motion_Prediction_CVPR_2022_paper.pdf) published in CVPR 2022.

![](assets/overview.png)

## Gettting Started

1\. Clone this repository:
```
git clone https://github.com/ZikangZhou/HiVT.git
cd HiVT
```

2\. Create a conda environment and install the dependencies:
```
conda create -n HiVT python=3.8
conda activate HiVT
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pytorch-geometric==1.7.2 -c rusty1s -c conda-forge
conda install pytorch-lightning==1.5.2 -c conda-forge
```

3\. Download [Argoverse Motion Forecasting Dataset v1.1](https://www.argoverse.org/av1.html). After downloading and extracting the tar.gz files, the dataset directory should be organized as follows:
```
/path/to/dataset_root/
├── train/
|   └── data/
|       ├── 1.csv
|       ├── 2.csv
|       ├── ...
└── val/
    └── data/
        ├── 1.csv
        ├── 2.csv
        ├── ...
```

4\. Install [Argoverse 1 API](https://github.com/argoai/argoverse-api).

## Training

To train HiVT-64:
```
python train.py --root /path/to/dataset_root/ --embed_dim 64
/home/manyazog/argoverse
```

To train HiVT-128:
```
python train.py --root /path/to/dataset_root/ --embed_dim 128
python train.py --root --root /home/manyazog/argoverse --embed_dim 128
```

**Note**: When running the training script for the first time, it will take several hours to preprocess the data (~3.5 hours on my machine). Training on an RTX 2080 Ti GPU takes 35-40 minutes per epoch.

During training, the checkpoints will be saved in `lightning_logs/` automatically. To monitor the training process:
```
tensorboard --logdir lightning_logs/
```

## Evaluation

To evaluate the prediction performance:
```
python eval.py --root /path/to/dataset_root/ --batch_size 32 --ckpt_path /path/to/your_checkpoint.ckpt
python eval.py --root /home/manyazog/argoverse --batch_size 32 --ckpt_path /home/manyazog/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt



 python eval.py --root /home/manyazog/argoverse --batch_size 32 --ckpt_path /home/manyazog/HiVT/lightning_logs/version_0/checkpoints/epoch=63-step=411903.ckpt
```

## Data Preprocessing

HiVT preprocesses each CSV scene into a serialized PyTorch Geometric sample (`.pt`) the first time you run training or evaluation.

- Preprocessing is triggered automatically when `ArgoverseV1Dataset` is constructed in `train.py` or `eval.py`.
- Raw files are read from:
  - `/path/to/dataset_root/train/data/`
  - `/path/to/dataset_root/val/data/`
- Processed files are cached at:
  - `/path/to/dataset_root/train/processed/`
  - `/path/to/dataset_root/val/processed/`

Each processed sample contains actor history/future tensors, masks, lane features, graph edges, and scene normalization metadata.

**Important**: The first preprocessing pass can take hours depending on your machine and storage speed. Later runs reuse the cached `.pt` files.

## Inference Pipeline

Inference in this repo is executed through the evaluation entry point (`eval.py`).

1. Load checkpoint and model hyperparameters:
  - `HiVT.load_from_checkpoint(..., parallel=True)`
2. Build validation dataset:
  - `ArgoverseV1Dataset(root=..., split='val', local_radius=model.hparams.local_radius)`
  - This step also triggers preprocessing if cache is missing.
3. Build dataloader:
  - `torch_geometric.data.DataLoader(..., shuffle=False)`
4. Run validation loop with PyTorch Lightning:
  - `trainer.validate(model, dataloader)`
5. Aggregate and report inference metrics:
  - `val_minADE`, `val_minFDE`, `val_minMR`

Quick command:

```
python eval.py --root /path/to/dataset_root/ --batch_size 32 --ckpt_path /path/to/your_checkpoint.ckpt
```

If you evaluate multiple checkpoints on the same dataset root, preprocessing is not repeated unless you delete the `processed/` directories.

## Visualization

To visualize one validation sample with map context, history, ground truth, and multimodal predictions:

```
python visualize.py --root /path/to/dataset_root/ --ckpt_path /path/to/your_checkpoint.ckpt --sample_idx 0
```

Example:

```
python visualize.py --root /home/manyazog/argoverse --ckpt_path /home/manyazog/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt --sample_idx 0
```

Useful options:

- `--map_radius`: lane query radius (meters) around scene origin, default is `80.0`.
- `--save_path`: save figure to disk, for example `--save_path vis_sample0.png`.

Plot legend:

- Gray lines: lane centerlines from ArgoverseMap.
- Blue line with circles: observed history.
- Green line with squares: ground-truth future.
- Red lines with triangles: predicted modes (alpha weighted by mode confidence).

## Pretrained Models

We provide the pretrained HiVT-64 and HiVT-128 in [checkpoints/](checkpoints). You can evaluate the pretrained models using the aforementioned evaluation command, or have a look at the training process via TensorBoard:
```
tensorboard --logdir checkpoints/
```

## Results

### Quantitative Results

For this repository, the expected performance on Argoverse 1.1 validation set is:

| Models | minADE | minFDE | MR |
| :--- | :---: | :---: | :---: |
| HiVT-64 | 0.69 | 1.03 | 0.10 |
| HiVT-128 | 0.66 | 0.97 | 0.09 |

### Qualitative Results

![](assets/visualization.png)

## Citation

If you found this repository useful, please consider citing our work:

```
@inproceedings{zhou2022hivt,
  title={HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction},
  author={Zhou, Zikang and Ye, Luyao and Wang, Jianping and Wu, Kui and Lu, Kejie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

## License

This repository is licensed under [Apache 2.0](LICENSE).

