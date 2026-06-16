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
"""Argoverse probabilistic ('p-') displacement / miss-rate metrics.

Exact match to argoverse.evaluation.eval_forecasting
(get_displacement_errors_and_miss_rate), with p_best = the (normalized)
probability of the mode that achieves the minimum displacement:

    p-minADE = minADE + min(-log(p_best), -log(0.05))
    p-minFDE = minFDE + min(-log(p_best), -log(0.05))
    p-MR     = mean[ 1.0 if minFDE > miss_threshold else (1 - p_best) ]

The -log(0.05) cap bounds the penalty for low-confidence predictions. Unlike the
Brier penalty (1 - p_best)^2, the log penalty is the NLL of the chosen mode, so
p-min* is the metric most directly aligned with an NLL-based distillation loss.
"""
import math
from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric

# -log(0.05); the official LOW_PROB_THRESHOLD_FOR_METRICS penalty cap.
_LOG_PROB_CAP = -math.log(0.05)


class PMinFDE(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(PMinFDE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                      process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob_best: torch.Tensor) -> None:
        fde = torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1)
        penalty = torch.clamp(-torch.log(prob_best.clamp_min(1e-12)), max=_LOG_PROB_CAP)
        self.sum += (fde + penalty).sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class PMinADE(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(PMinADE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                      process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob_best: torch.Tensor) -> None:
        ade = torch.norm(pred - target, p=2, dim=-1).mean(dim=-1)
        penalty = torch.clamp(-torch.log(prob_best.clamp_min(1e-12)), max=_LOG_PROB_CAP)
        self.sum += (ade + penalty).sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class PMR(Metric):

    def __init__(self,
                 miss_threshold: float = 2.0,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(PMR, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                  process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.miss_threshold = miss_threshold

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob_best: torch.Tensor) -> None:
        fde = torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1)
        miss = fde > self.miss_threshold
        # Hits contribute (1 - p_best); misses contribute the full 1.0.
        contrib = torch.where(miss, torch.ones_like(prob_best), 1.0 - prob_best)
        self.sum += contrib.sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
