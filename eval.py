# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset
from models.hivt import HiVT

# PyTorch >= 2.6 defaults torch.load to weights_only=True, which fails on
# checkpoints that pickle objects like the ModelCheckpoint callback. These are
# our own trusted checkpoints, so restore the old behaviour.
_orig_torch_load = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args)

    # KD checkpoints are trained with HiVTKD, which wraps a HiVT as `self.student`,
    # so every weight is saved under a `student.` prefix. The plain HiVT class
    # expects un-prefixed keys. Detect this case, strip the prefix, and load the
    # student weights into a plain HiVT — this runs the IDENTICAL validation path
    # used for the HiVT-32/64/128 baselines (same oracle min-over-modes), so the
    # numbers are directly comparable.
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    is_kd = any(k.startswith('student.') for k in state_dict)
    if is_kd:
        hp = dict(ckpt['hyper_parameters'])
        # Drop KD-only / Trainer hparams that HiVT.__init__ does not accept.
        hivt_keys = {
            'historical_steps', 'future_steps', 'num_modes', 'rotate', 'node_dim',
            'edge_dim', 'embed_dim', 'num_heads', 'dropout', 'num_temporal_layers',
            'num_global_layers', 'local_radius', 'parallel', 'lr', 'weight_decay',
            'T_max',
        }
        hivt_kwargs = {k: v for k, v in hp.items() if k in hivt_keys}
        hivt_kwargs['parallel'] = True
        model = HiVT(**hivt_kwargs)
        student_sd = {k[len('student.'):]: v for k, v in state_dict.items()
                      if k.startswith('student.')}
        model.load_state_dict(student_sd, strict=True)
        model.eval()
    else:
        model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=True)
    val_dataset = ArgoverseV1Dataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer.validate(model, dataloader)
