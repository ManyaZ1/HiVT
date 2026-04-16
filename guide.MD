# HiVT — Codemap & Guide

This document explains the structure of the HiVT implementation and walks through the main pieces implemented in the repository. Focus is on the core training model defined in [`models/hivt.py`](models/hivt.py) — see [`models.HiVT`](models/hivt.py).

## Core model: HiVT
- File: [models/hivt.py](models/hivt.py)  
  Class: [`models.HiVT`](models/hivt.py)  
  Key methods:
  - [`models.HiVT.__init__`](models/hivt.py) — constructs submodules and losses.
  - [`models.HiVT.forward`](models/hivt.py) — the inference path that produces predictions.
  - [`models.HiVT.training_step`](models/hivt.py) — training loss computation and logging.
  - [`models.HiVT.validation_step`](models/hivt.py) — validation losses and metric updates.
  - [`models.HiVT.configure_optimizers`](models/hivt.py) — optimizer + scheduler setup.
  - [`models.HiVT.add_model_specific_args`](models/hivt.py) — CLI args for the model.

### What HiVT composes
- Local encoder: [`models.LocalEncoder`](models/local_encoder.py) — implemented in [models/local_encoder.py](models/local_encoder.py). Produces per-node local embeddings from historical trajectories.
- Global interactor: [`models.GlobalInteractor`](models/global_interactor.py) — implemented in [models/global_interactor.py](models/global_interactor.py). Aggregates information across nodes to produce global context.
- Decoder: [`models.MLPDecoder`](models/decoder.py) — implemented in [models/decoder.py](models/decoder.py). Takes local + global embeddings and outputs multimodal future trajectories and mode probabilities.

### Losses and metrics
- Regression (aleatoric) loss: [`losses.LaplaceNLLLoss`](losses/laplace_nll_loss.py) — [losses/laplace_nll_loss.py](losses/laplace_nll_loss.py)
- Classification loss for modes: [`losses.SoftTargetCrossEntropyLoss`](losses/soft_target_cross_entropy_loss.py) — [losses/soft_target_cross_entropy_loss.py](losses/soft_target_cross_entropy_loss.py)
- Evaluation metrics: [`metrics.ADE`](metrics/ade.py), [`metrics.FDE`](metrics/fde.py), [`metrics.MR`](metrics/mr.py) — files: [metrics/ade.py](metrics/ade.py), [metrics/fde.py](metrics/fde.py), [metrics/mr.py](metrics/mr.py)

### Data struct used by the model
- The model expects a `TemporalData`-like object: see [`utils.TemporalData`](utils.py) — file: [utils.py](utils.py). The forward and steps access fields such as:
  - `data.y` — ground-truth future trajectories
  - `data['padding_mask']` — used to mask padded future frames
  - `data['rotate_angles']` / `data['rotate_mat']` — rotation augment handling
  - `data.num_nodes`, `data.num_graphs`, `data['agent_index']` — indexing used in training/validation

## How forward / training / validation work (brief)
- Forward (`models.HiVT.forward`):
  - Optional rotation: builds per-node 2×2 rotation matrices and rotates `data.y` if present.
  - Calls [`models.LocalEncoder`](models/local_encoder.py) → local embeddings.
  - Calls [`models.GlobalInteractor`](models/global_interactor.py) with local embeddings → global embeddings.
  - Calls [`models.MLPDecoder`](models/decoder.py) → returns `y_hat` (F × N × T × D) and `pi` (mode probs).
- Training step (`models.HiVT.training_step`):
  - Builds `reg_mask` from `data['padding_mask']` to identify valid future frames.
  - Computes per-mode L2 norms between predicted modes and ground truth, picks `best_mode`.
  - Computes regression loss with [`losses.LaplaceNLLLoss`](losses/laplace_nll_loss.py) on best-mode predictions.
  - Builds soft targets from per-mode errors and computes classification loss with [`losses.SoftTargetCrossEntropyLoss`](losses/soft_target_cross_entropy_loss.py).
  - Logs `train_reg_loss` and returns combined loss.
- Validation step (`models.HiVT.validation_step`):
  - Similar regression loss computation and logging.
  - Computes agent-centric best-mode predictions and updates minADE/minFDE/minMR metrics via [`metrics.ADE`](metrics/ade.py), [`metrics.FDE`](metrics/fde.py), [`metrics.MR`](metrics/mr.py).

## Optimizer setup
- Implemented in [`models.HiVT.configure_optimizers`](models/hivt.py).
- Groups parameters into "decay" and "no_decay" sets using type-based whitelist/blacklist (Linear, Conv, LayerNorm, Embedding, BatchNorm, etc.), then builds AdamW optimizer and a CosineAnnealingLR scheduler.

## CLI and training scripts
- Model CLI args are exposed via [`models.HiVT.add_model_specific_args`](models/hivt.py).
- Entry points:
  - Training: [train.py](train.py)
  - Evaluation: [eval.py](eval.py)
  - Dataset / datamodule:
    - [`datamodules.argoverse_v1_datamodule`](datamodules/argoverse_v1_datamodule.py)
    - [`datasets.argoverse_v1_dataset`](datasets/argoverse_v1_dataset.py)

## File map (quick)
- Model code:
  - [models/hivt.py](models/hivt.py) — main LightningModule (`models.HiVT`)
  - [models/local_encoder.py](models/local_encoder.py) — local temporal encoder (`models.LocalEncoder`)
  - [models/global_interactor.py](models/global_interactor.py) — global interaction (`models.GlobalInteractor`)
  - [models/decoder.py](models/decoder.py) — MLP decoder (`models.MLPDecoder`)
  - [models/embedding.py](models/embedding.py) — embedding helpers
- Losses:
  - [losses/laplace_nll_loss.py](losses/laplace_nll_loss.py)
  - [losses/soft_target_cross_entropy_loss.py](losses/soft_target_cross_entropy_loss.py)
- Metrics:
  - [metrics/ade.py](metrics/ade.py)
  - [metrics/fde.py](metrics/fde.py)
  - [metrics/mr.py](metrics/mr.py)
- Data & IO:
  - [utils.py](utils.py) — `TemporalData` and helpers
  - [datamodules/argoverse_v1_datamodule.py](datamodules/argoverse_v1_datamodule.py)
  - [datasets/argoverse_v1_dataset.py](datasets/argoverse_v1_dataset.py)
- Project entry points / docs:
  - [train.py](train.py)
  - [eval.py](eval.py)
  - [README.md](README.md)

## Recommended next steps to explore
1. Open the forward path: [`models.HiVT.forward`](models/hivt.py) in [models/hivt.py](models/hivt.py) to see how data flows.
2. Inspect [`models.LocalEncoder`](models/local_encoder.py) and [`models.GlobalInteractor`](models/global_interactor.py) to understand the representation pipeline.
3. Read the decoder implementation: [`models.MLPDecoder`](models/decoder.py) to see prediction head and output format.
4. Review the loss implementations in [losses/](losses) and metric implementations in [metrics/](metrics).
5. Run training via [train.py](train.py) and monitor logs to match code behavior.

If you want, I can generate a ready-to-add markdown file (eg., docs/HIVT_OVERVIEW.md) with this content.

## Paper Module Mapping (6-part architecture)

1. Scene representation + local partition around each agent
  - Scene construction and normalization happen in [datasets/argoverse_v1_dataset.py](datasets/argoverse_v1_dataset.py#L90).
  - Agent-centric origin/rotation is created in [datasets/argoverse_v1_dataset.py](datasets/argoverse_v1_dataset.py#L111).
  - Full actor graph is initialized in [datasets/argoverse_v1_dataset.py](datasets/argoverse_v1_dataset.py#L119).
  - Lane features and lane-actor edges are built in [datasets/argoverse_v1_dataset.py](datasets/argoverse_v1_dataset.py#L156).
  - Nearby lane selection by radius is in [datasets/argoverse_v1_dataset.py](datasets/argoverse_v1_dataset.py#L197) and [datasets/argoverse_v1_dataset.py](datasets/argoverse_v1_dataset.py#L228).
  - Runtime local distance filtering uses `DistanceDropEdge` in [utils.py](utils.py#L61) and is applied from [models/local_encoder.py](models/local_encoder.py#L51).

2. Agent-agent interaction module
  - Implemented as `AAEncoder` in [models/local_encoder.py](models/local_encoder.py#L100).
  - Attention message passing is defined in [models/local_encoder.py](models/local_encoder.py#L169).

3. Temporal dependency module
  - Implemented as `TemporalEncoder` in [models/local_encoder.py](models/local_encoder.py#L218).
  - Transformer layer implementation is in [models/local_encoder.py](models/local_encoder.py#L257).

4. Agent-lane interaction module
  - Implemented as `ALEncoder` in [models/local_encoder.py](models/local_encoder.py#L294).
  - Lane-to-actor attention message logic is in [models/local_encoder.py](models/local_encoder.py#L351).

5. Global interaction module
  - Implemented as `GlobalInteractor` in [models/global_interactor.py](models/global_interactor.py#L31).
  - Stacked global interaction layers are created in [models/global_interactor.py](models/global_interactor.py#L51).
  - Per-layer message passing is implemented in [models/global_interactor.py](models/global_interactor.py#L81).

6. Multimodal decoder
  - Main decoder used by HiVT is `MLPDecoder` in [models/decoder.py](models/decoder.py#L87).
  - Trajectory heads are defined in [models/decoder.py](models/decoder.py#L108).
  - Mode probability head (`pi`) is defined in [models/decoder.py](models/decoder.py#L119).
  - Decoder is wired into HiVT in [models/hivt.py](models/hivt.py#L78).

End-to-end wiring in model forward:
- Local encoder -> global interactor -> decoder in [models/hivt.py](models/hivt.py#L90).
