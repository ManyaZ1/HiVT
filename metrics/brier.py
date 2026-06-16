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
"""Argoverse probability-aware displacement metrics.

brier-minFDE / brier-minADE augment the (oracle) min-over-modes displacement
error with a Brier penalty on the probability assigned to the chosen mode:

    brier-minFDE = minFDE + (1 - p_best)^2
    brier-minADE = minADE + (1 - p_best)^2

where p_best is the predicted probability of the mode that achieves the minimum
displacement. Unlike plain minADE/minFDE (which ignore the mixture weights),
these reward putting confidence on the mode that turns out best, so they are
sensitive to how well a distillation step transfers the teacher's mode weights.
"""
from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class BrierFDE(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(BrierFDE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                       process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob_best: torch.Tensor) -> None:
        fde = torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1)
        self.sum += (fde + (1.0 - prob_best) ** 2).sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class BrierADE(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(BrierADE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                       process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor,
               prob_best: torch.Tensor) -> None:
        ade = torch.norm(pred - target, p=2, dim=-1).mean(dim=-1)
        self.sum += (ade + (1.0 - prob_best) ** 2).sum()
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
