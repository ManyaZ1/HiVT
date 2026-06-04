"""
hivt_kd_patch_notes.py
======================
This file is NOT runnable — it documents exactly what to change in hivt_kd.py
to fix the teacher/student shape mismatch once diagnose_shapes.py confirms
the teacher tensors have the right shape out of KDDataset.

CONTEXT
-------
After the library upgrade, check_collation.py shows:

    Single item teacher_loc shape: torch.Size([1, 6, 30, 2])   ← BUG: extra leading dim
    Batched teacher_loc shape:     torch.Size([2, 6, 30, 2])
    Needs permute? True

Two separate problems:

Problem A — Extra leading dim in single item  [1, 6, 30, 2] instead of [6, 30, 2]
---------------------------------------------------------------------------
This comes from TeacherStore (or the save path).  Fix is in
kd_teacher_store_fixed.py (provided).  After the fix:

    Single item teacher_loc: [6, 30, 2]
    Batched teacher_loc:     [B, 6, 30, 2]

Problem B — PyG collation dimension ordering
---------------------------------------------------------------------------
Old PyG stacked unknown tensor attributes along a new dim-0 → [B, ...].
New PyG does the same for plain tensors, UNLESS the attribute is registered
in the Data schema. So [6,30,2] per item → [B,6,30,2] is now correct.

But HiVT decoder outputs:
    loc_pred   : [F, N_total, H, 2]   where F=6, N_total = all agents in batch
    pi_pred    : [N_total, F]

Teacher cache stores only the FOCAL agent per scene:
    teacher_loc : [F, H, 2] per scene → collated [B, F, H, 2]

So the shapes need to be reconciled in training_step.

─────────────────────────────────────────────────────────────────
PATCH for hivt_kd.py   training_step
─────────────────────────────────────────────────────────────────
"""

# ── BEFORE (broken) ──────────────────────────────────────────────────────────
def training_step_BROKEN(self, data, batch_idx):
    # ... model forward ...
    loc_pred, scale_pred, pi_pred = self.decoder(...)  # [F,N_total,H,2], [F,N_total,H,2], [N_total,F]

    # BUG: teacher tensors from batch are [B,F,H,2] but loss expects [F,N,H,2]
    # No permute, no focal-agent slicing → complete shape mismatch
    kd_loss, kd_metrics = self.kd_loss_fn(
        loc_pred, scale_pred, pi_pred,
        data.teacher_loc, data.teacher_scale, data.teacher_pi,
    )


# ── AFTER (correct) ──────────────────────────────────────────────────────────
def training_step_FIXED(self, data, batch_idx):
    # ... model forward ...
    loc_pred, scale_pred, pi_pred = self.decoder(...)  # [F,N_total,H,2], [N_total,F]

    # ── Step 1: permute teacher tensors from collation convention to loss convention ──
    # Collated by PyG:  [B, F, H, 2]
    # kd_loss expects:  [F, B, H, 2]   (same convention as HiVT decoder)
    loc_t   = data.teacher_loc.permute(1, 0, 2, 3)    # [B,F,H,2] → [F,B,H,2]
    scale_t = data.teacher_scale.permute(1, 0, 2, 3)  # same
    pi_t    = data.teacher_pi                           # [B,F] — already correct for kd_loss

    # ── Step 2: slice student outputs to focal agents only ────────────────────
    # data.agent_index holds, for each graph in the batch, the index of the
    # focal (target) agent within N_total.  HiVT calls this `data.av_index`
    # or similar depending on your dataset version.
    #
    # The exact attribute name:  check what ArgoverseV1Dataset stores.
    # Commonly: data.av_index (shape [B]) or derived from data.batch + data.rotate_imat
    #
    # Option 1 — if your dataset stores data.av_index:
    focal_idx = data.av_index                  # [B]  indices into N_total
    loc_s_focal   = loc_pred[:, focal_idx, :, :]    # [F, B, H, 2]
    scale_s_focal = scale_pred[:, focal_idx, :, :]  # [F, B, H, 2]
    pi_s_focal    = pi_pred[focal_idx, :]            # [B, F]

    # Option 2 — if each graph has exactly 1 focal agent and they appear at
    # a known position (e.g. index 0 of each sub-graph after normalization),
    # you can reconstruct focal_idx from data.ptr or data.batch:
    #   focal_idx = data.ptr[:-1]   # first node of each graph
    # This is fragile; prefer Option 1 if av_index is available.

    # ── Step 3: guard against missing teacher data ────────────────────────────
    if hasattr(data, "has_teacher") and not data.has_teacher.all():
        # filter to scenes that have teacher data
        mask = data.has_teacher.bool()
        loc_s_focal   = loc_s_focal[:, mask, :, :]
        scale_s_focal = scale_s_focal[:, mask, :, :]
        pi_s_focal    = pi_s_focal[mask, :]
        loc_t         = loc_t[:, mask, :, :]
        scale_t       = scale_t[:, mask, :, :]
        pi_t          = pi_t[mask, :]

    if loc_s_focal.shape[1] == 0:
        # entire batch had no teacher data — skip KD loss
        kd_loss, kd_metrics = torch.tensor(0.0), {}
    else:
        kd_loss, kd_metrics = self.kd_loss_fn(
            loc_s_focal, scale_s_focal, pi_s_focal,
            loc_t.detach(), scale_t.detach(), pi_t.detach(),
        )

    # ... compute task loss, combine, return ...


"""
─────────────────────────────────────────────────────────────────
VERIFICATION CHECKLIST before running smoke_train.py
─────────────────────────────────────────────────────────────────

[ ]  diagnose_shapes.py  →  all PASS
     - single item teacher_loc shape = [6, 30, 2]   (no leading 1)
     - batched teacher_loc shape     = [B, 6, 30, 2]

[ ]  kd_teacher_store_fixed.py in place of kd_teacher_store.py

[ ]  hivt_kd.py training_step uses permute + focal slicing as above

[ ]  Confirm data.av_index (or equivalent) exists:
         python -c "
         from datasets import ArgoverseV1Dataset
         d = ArgoverseV1Dataset(root='...', split='train')
         item = d[0]
         print([k for k in item.keys()])
         "

[ ]  smoke_train.py runs to completion with all PASS

[ ]  Optionally: run with embed_dim=128 (teacher size) and check that
     KD loss converges faster/lower than with embed_dim=64, confirming
     the teacher signal is meaningful.
─────────────────────────────────────────────────────────────────
"""
