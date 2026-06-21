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
"""Diagnostic calibration / sharpness metrics for the predicted Laplace.

These quantify the KD over-confidence mechanism documented in
docs/kd_emb32_kl_comparison.md: the v1 mean-target loss shrinks the student's
Laplace scales, so a more-distilled student should show smaller `b` (sharper)
and worse coverage (under-covered) than a less-distilled one.

All states are non-persistent (torchmetrics `add_state` default), so adding
these metrics to HiVT does NOT add keys to the model state_dict — eval.py still
loads stripped KD checkpoints with strict=True.
"""
from typing import Any, Callable, Optional, Sequence

import torch
from torchmetrics import Metric

# Nominal central-interval coverage levels for the reliability curve.
_DEFAULT_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class ScalarMean(Metric):
    """Running mean of every element of the values passed to `update`.

    A tiny element-weighted mean (sum / count) so epoch aggregation across
    batches of different sizes is correct, matching the repo's metric pattern.
    """

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(ScalarMean, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                         process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, value: torch.Tensor) -> None:
        value = value.detach()
        self.sum += value.sum()
        self.count += value.numel()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class LaplaceCoverage(Metric):
    """Empirical central-interval coverage of a predicted Laplace(mu, b).

    For a Laplace, P(|X - mu| <= t) = 1 - exp(-t/b), so the central interval
    with nominal coverage p has half-width t(p) = -b * ln(1 - p). For each
    nominal level p we accumulate, over all valid (agent, timestep, coord)
    points, the number whose error |y - mu| falls inside t(p), plus the total
    count. `compute()` returns the empirical coverage per level; a perfectly
    calibrated model has empirical(p) == p, and under-coverage (empirical < p)
    means the predicted scales are too small (over-confident).
    """

    def __init__(self,
                 levels: Sequence[float] = _DEFAULT_LEVELS,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(LaplaceCoverage, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                              process_group=process_group, dist_sync_fn=dist_sync_fn)
        # persistent=False keeps this constant out of the model state_dict.
        self.register_buffer('levels', torch.tensor(list(levels), dtype=torch.float), persistent=False)
        self.add_state('hits', default=torch.zeros(len(levels)), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self,
               loc: torch.Tensor,
               scale: torch.Tensor,
               target: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> None:
        # loc / scale / target: [..., 2]; mask broadcasts over the last (coord) dim.
        err = (target - loc).abs()
        # Half-width per level: t(p) = -b * ln(1 - p). [L, *err.shape]
        log_term = (-torch.log1p(-self.levels)).to(err)            # [L]
        t = scale.unsqueeze(0) * log_term.view(-1, *([1] * err.dim()))
        inside = err.unsqueeze(0) <= t                             # [L, *err.shape]
        if mask is not None:
            m = mask.unsqueeze(-1).expand_as(err)                  # [*err.shape]
            inside = inside & m.unsqueeze(0)
            self.count += m.sum()
        else:
            self.count += err.numel()
        self.hits += inside.flatten(1).sum(dim=1).to(self.hits)

    def compute(self) -> torch.Tensor:
        return self.hits / self.count

    def calibration_error(self) -> torch.Tensor:
        """Scalar ECE-like miscalibration: mean_p |empirical(p) - p|."""
        cov = self.compute()
        return (cov - self.levels.to(cov)).abs().mean()


def log_laplace_coverage(module, coverage: LaplaceCoverage, prog_bar: bool = False) -> None:
    """Log the full reliability curve (`val_cov_pXX`) + scalar `val_calib_err`.

    Called from `on_validation_epoch_end` because the per-level coverage needs
    the globally-accumulated hit/count state (a per-batch mean would be wrong).
    Logs via `module.log` so the KD wrapper logs to its own W&B run while reusing
    the coverage metric that lives on `module.student`.
    """
    if float(coverage.count) <= 0:
        return
    cov = coverage.compute()
    levels = coverage.levels.to(cov)
    for p, c in zip(levels.tolist(), cov.tolist()):
        module.log(f'val_cov_p{int(round(p * 100)):02d}', c, on_step=False, on_epoch=True, batch_size=1)
    module.log('val_calib_err', (cov - levels).abs().mean(), prog_bar=prog_bar,
               on_step=False, on_epoch=True, batch_size=1)
    coverage.reset()
