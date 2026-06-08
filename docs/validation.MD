# HiVT training

## 32 trained for 64 epochs
```bash
python eval.py --root /home/manya/argoverse --batch_size 32 --ckpt_path /home/manya/HiVT/HiVT-32/gxhl2ug9/checkpoints/epoch=63-step=411903.ckpt
--------------------------------------------------------------------------------
DATALOADER:0 VALIDATE RESULTS
{'val_minADE': 0.744571328163147,
 'val_minFDE': 1.1781848669052124,
 'val_minMR': 0.12505066394805908,
 'val_reg_loss': -0.23460741341114044}
--------------------------------------------------------------------------------
```
## 64 trained for 64 epochs
```bash
python eval.py --root /home/manya/argoverse --batch_size 32 --ckpt_path /home/manya/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt
--------------------------------------------------------------------------------
DATALOADER:0 VALIDATE RESULTS
{'val_minADE': 0.6868845820426941,
 'val_minFDE': 1.0300804376602173,
 'val_minMR': 0.10262971371412277,
 'val_reg_loss': -0.30333220958709717}
--------------------------------------------------------------------------------
```
# 128 trained for 128 epochs
```bash
python eval.py --root /home/manya/argoverse --batch_size 32 --ckpt_path /home/manya/HiVT/checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt
--------------------------------------------------------------------------------
DATALOADER:0 VALIDATE RESULTS
{'val_minADE': 0.6611008048057556,
 'val_minFDE': 0.9691530466079712,
 'val_minMR': 0.09203992784023285,
 'val_reg_loss': -0.33467531204223633}
--------------------------------------------------------------------------------
```
# config.py  — single source of truth for all hyperparameters
config = {
    # Model
    'hidden_dim':       64,       # your current setting
    'num_heads':        8,
    'num_layers':       4,
    # Training
    'lr':               1e-3,
    'batch_size':       32,
    'epochs':           64,
    # Robustness
    'mask_ratio_min':   0.1,
    'mask_ratio_max':   0.6,
    'lambda_distill':   0.5,
    'noise_type':       'suffix_mask',
    # Eval
    'eval_mask_ratios': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}

# In your main training script:
import wandb
wandb.init(
    project='hivt-robustness',
    name='distill-suffix-lambda05',   # descriptive, not 'run_3'
    config=config,
)

# Log every step
wandb.log({
    'train/loss_total':   losses['loss_total'],
    'train/loss_full':    losses['loss_full'],
    'train/loss_distill': losses['loss_distill'],
    'train/mask_ratio':   losses['mask_ratio'],
    'epoch': epoch,
})

# Log validation metrics after each epoch
wandb.log({
    'val/minADE6_clean':       metrics_clean['minADE6'],
    'val/minFDE6_clean':       metrics_clean['minFDE6'],
    'val/MR6_clean':           metrics_clean['MR6'],
    'val/minFDE6_mask40':      metrics_mask40['minFDE6'],
    'val/MR6_mask40':          metrics_mask40['MR6'],
})