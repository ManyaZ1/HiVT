python eval.py --root /home/manya/argoverse --batch_size 32 --ckpt_path /home/manya/HiVT/HiVT-32/gxhl2ug9/checkpoints/epoch=63-step=411903.ckpt
Global seed set to 2022
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
/home/manya/miniconda3/envs/hivt_new/lib/python3.10/site-packages/pytorch_lightning/core/lightning.py:2065: DeprecationWarning: `torch.distributed._sharded_tensor` will be deprecated, use `torch.distributed._shard.sharded_tensor` instead
  from torch.distributed._sharded_tensor import pre_load_state_dict_hook, state_dict_hook
/home/manya/HiVT/models/local_encoder.py:228: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer was not TransformerEncoderLayer
  self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
/home/manya/HiVT/eval.py:38: UserWarning: 'data.DataLoader' is deprecated, use 'loader.DataLoader' instead
  dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Validating:   0%|                                                                                                                  | 0/1234 [00:00<?, ?it/s]/home/manya/miniconda3/envs/hivt_new/lib/python3.10/site-packages/pytorch_lightning/utilities/data.py:56: UserWarning: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 1. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
  warning_cache.warn(
Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉| 1233/1234 [01:01<00:00, 25.26it/s]--------------------------------------------------------------------------------
DATALOADER:0 VALIDATE RESULTS
{'val_minADE': 0.744571328163147,
 'val_minFDE': 1.1781848669052124,
 'val_minMR': 0.12505066394805908,
 'val_reg_loss': -0.23460741341114044}
--------------------------------------------------------------------------------