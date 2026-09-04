# Handoff — ICRA video preparation

State of the conference-video plan for *Permutation-Invariant Knowledge Distillation for
Motion Prediction in Autonomous Vehicles* ([drivex2026_kd.tex](drivex2026_kd.tex)).
A fresh conversation can resume from here.

**Scope of this document:** the *framing* decisions (what the paper claims and how the video
argues it) and the *technical facts* needed to build the video's hero visual. It does not
duplicate the scientific report of record, [kd_emb32_kl_comparison.md](kd_emb32_kl_comparison.md),
or the operational state, [handoff_kd_thesis_state.md](handoff_kd_thesis_state.md).

---

## 1. Framing — settled decisions

### 1.1 What the contribution is NOT

Two framings were considered and **rejected**:

- ~~"We compressed HiVT with knowledge distillation."~~ Distilling a forecaster is an
  engineering exercise. This framing invites reject.
- ~~"Examining knowledge distillation on trajectory forecasters."~~ A survey framing.
  "Examining" promises an investigation with no target and no success criterion. The paper
  has a method and two proved properties; it should claim them.

Also rejected as a *narrative*: "method 1 had flaws, method 2 fixed them." That describes the
chronology of the work, not a result. v1 is **not a strawman** — it is what any competent
person writes down first, and v2 is the principled object of which v1 is the zero-variance
limit (Property 2).

### 1.2 What the contribution IS

> **Distilling a mixture-density forecaster is a structurally different problem than
> distilling a classifier, and the obvious objective is quietly wrong.**

Three parts, in decreasing order of strength:

1. **A diagnosis nobody had quantified.** WTA training leaves mode *index* semantically
   meaningless, so the standard KD move — match output *k* to output *k* — compares unrelated
   things. Measured: the identity pairing is optimal in **0% of scenes**.
2. **An objective that removes the problem instead of patching it.** Not Hungarian matching
   (hard, non-differentiable, undefined for K_S ≠ K_T, flips discontinuously between steps) —
   score teacher modes under the student's *entire* mixture, so matching never happens.
3. **A pathology with a closed-form explanation and a principled fix.** Eq. (8) shows the
   stationary scale is a weighted average of the student's own residuals; the teacher's spread
   appears **nowhere**. v2 is exactly the forward KL; v1 is its zero-variance limit.

**Objective statement to use:**

> Derive a distillation objective that is correct for multi-modal forecasters — invariant to
> how each model orders its modes, and faithful to the uncertainty each mode carries.

### 1.3 The sharpest sentence (not yet in the paper — add it)

v1 and v2 tie on minADE, minFDE, and MR. Therefore:

> **A field that reports only best-of-K geometry cannot see this failure at all.**

Someone shipping v1 would see clean numbers and a model that lies about its uncertainty.

### 1.4 Impact, three levels

- **Measured.** A 46k-param student reaches within 5.8% of an undistilled model with 3.7× the
  parameters (~83% of a size class recovered), with calibration *better* than both the
  from-scratch baseline and the teacher. 3 seeds, non-overlapping clouds. Free at inference —
  training-time loss only, teacher cached offline.
- **Transferable.** Needs only a mixture head + WTA training, i.e. nearly every modern
  forecaster. *If your forecaster emits a WTA-trained mixture, index-aligned distillation is
  wrong for your model too.*
- **The warning.** Compression is assumed to degrade accuracy gracefully and uncertainty
  invisibly. The naive route actively destroys calibration while looking fine on the
  leaderboard; the right route preserves it at no cost.

### 1.5 Vulnerability to pre-empt

The teacher is 2.56M parameters — small by modern standards. A reviewer can ask *"who needs to
compress that?"* Answer before they ask: the students are **46k**, genuinely
microcontroller-class, and the frontier argument (recovering a full size class at your own
parameter count) is scale-free in principle. A real embedded latency number settles it.

---

## 2. The video

### 2.1 ICRA constraints (verify against the current CFP)

- **3:00 hard cap**; MP4/H.264; PaperPlaza attachment limit ~25 MB → ~1.1 Mbps at 3 min.
  **1080p screen capture will not fit.** Render **1280×720**; flat-color matplotlib animations
  on white encode at a fraction of that budget.
- **Assume muted.** Burn in captions; no fact carried by audio alone.
- **Assume a 30-second skim.** Permutation figure and headline number both before 0:45.
- **Self-contained**: title/authors/affiliation card; must make sense without the PDF.

### 2.2 Two hero beats — both required

Neither works alone:

| Beat | Time | Establishes | Without it |
|---|---|---|---|
| **1. Permutation** | ~30 s | Why the *standard* approach is wrong (0% identity-optimal). The outward-facing claim. | Beat 2 alone reads as "we fixed our own bug" — an internal v1-vs-v2 comparison. |
| **2. Uncertainty montage** | ~35 s | Why the *obvious fix* is also wrong, and what it costs. Rigor + safety payoff. | Beat 1 alone is a diagnosis with a shallow remedy. |

Order matters: beat 1 earns the right to beat 2. Everything else (frontier plot, seed table,
per-horizon curve, embedded footprint) is evidence serving those two, and gets cut to fit.

### 2.3 The montage — design decisions

**Three panels, not two.** `from-scratch | v1 | v2`. A v1-vs-v2 split shows variant A against
variant B; adding no-KD makes it *"the model you'd ship today / what the obvious distillation
does to it / ours."* Same rendering work, different message.

**A single scene will not work.** v1's coverage is 0.711 — on a random scene its envelope still
contains the ground truth ~7 times in 10. One scene means either showing a case where nothing
visible is wrong, or hand-picking the 29% and passing it off as typical. A reviewer would be
right to object.

**Fix: montage with a running tally.** ~10–15 scenes at ~1.5 s each, three panels, with a
per-scene point counter and a cumulative percentage under each panel. The counters diverge as
it runs and land on **0.903 / 0.711 / 0.909**. This animates the reliability diagram out of raw
scenes instead of asserting it; cherry-picking becomes structurally impossible. Hold on one
final scene so the viewer can see the **36% width contraction**, which is visible on every
scene regardless of whether the GT falls inside.

**Scene selection must be by stated criterion**, not by eye (e.g. highest teacher mode-entropy
= genuinely ambiguous intersections), and the criterion goes on screen.

### 2.4 Honesty guardrail

Do **not** add a scripted "the ego car crashes because of overconfidence" animation. Planner
behaviour is not measured (the paper's own limitations say so) and a fabricated collision is
exactly what a reviewer punishes. Either do the planner-in-the-loop experiment, or label the
shot **"illustrative — no planner in the loop"** in visible text.

---

## 3. Technical facts for building the montage

### 3.1 Model output layout — confirmed

`y_hat` from [models/decoder.py:84](../models/decoder.py#L84) is **`[F, N, H, 4]`** =
`(loc_x, loc_y, b_x, b_y)`, `F=6` modes, `H=30` steps. `pi` is **`[N, F]` raw logits** → apply
`softmax`. Scales are `elu(·)+1+min_scale`, so strictly positive.

Both `y_hat` and `data.y` live in the **agent-local rotated frame** (`data.rotate_mat`).

### 3.2 The exact coverage statistic (must be reproduced, not approximated)

From [models/hivt.py:196-203](../models/hivt.py#L196-L203) and
[metrics/calibration.py](../metrics/calibration.py):

- **Focal agents only** — `data['agent_index']`, one per scene, not all agents.
- **Best mode selected by min FDE** (`best_mode_agent`), not the highest-π mode, not the mixture.
- Band half-width for nominal level *p*: **`t(p) = -b · ln(1 - p)`**. For p=0.90,
  `t = 2.3026 · b`.
- Counted **per (agent, timestep, coordinate)** point under `reg_mask` → **60 points per scene**
  (30 steps × 2 coords).
- `calib_err` = mean absolute gap between empirical and nominal coverage over
  levels `(0.1 … 0.9)`.

**Consequence for the visual:** the per-scene counter must read *"90% band contains 47/60
points"*, **not** a binary "GT inside / outside". Only the point-wise version accumulates to
0.711 / 0.909.

**Consequence for the frame:** compute the tally in the **agent-local frame** (where `b_x, b_y`
are axis-aligned and the metric is defined); rotate to the scene frame for **display only**.

### 3.3 Checkpoint map — HiVT-32 full data, the three montage arms

> **The table below is WRONG and is kept only so the error is recognisable.**
> The no-KD and v1 rows are checkpoints that [CHECKPOINTS.md](CHECKPOINTS.md)
> explicitly rules out (`emb32-bs128-lkl0.0` and `emb32-bs128-lkl0.5` are
> *different runs*, not the paper's arms). **Use CHECKPOINTS.md.**
> `precompute_montage.py` already wires the correct ones, which is why the
> rendered montage passes the gate.

| Arm | Checkpoint | Paper minFDE |
|---|---|---|
| from-scratch (no KD) | ~~`kd_ckpt/emb32-bs128-lkl0.0/best/HiVTKD-epoch=46-val_minFDE=1.23.ckpt`~~ | 1.157 |
| v1 (mean-target, λ=0.5) | ~~`kd_ckpt/emb32-bs128-lkl0.5/best/HiVTKD-epoch=62-val_minFDE=1.12.ckpt`~~ | 1.050 |
| v2 (dist-matching, λ=0.5) | `kd_ckpt/emb32-bs128-lkl0.5-distv2-full/best/HiVTKD-epoch=63-val_minFDE=1.05.ckpt` | 1.050 |

Teacher: `checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt`.

All three arms are **HiVTKD-wrapped** → every weight is under a `student.` prefix. Strip it and
load into a plain `HiVT`, exactly as [eval.py:53-60](../eval.py#L53-L60) does. Filenames carry
the in-training `val_minFDE`, which differs from the eval.py number — **the paper's numbers come
from eval.py**; see [[eval-ground-truth]].

### 3.4a RUNBOOK — copy-pasteable, no session needed

```bash
cd /home/manya/HiVT
source /home/manya/miniconda3/etc/profile.d/conda.sh && conda activate hivt_new
```

**Always use `python -u`.** `stdbuf -oL python` does NOT work — Python buffers stdout
itself, so a long run looks frozen for its entire duration and dumps everything at the
end. Use `-u` or you are flying blind.

| step | command | time |
|---|---|---|
| 1. full scan + select | `python -u -m KD.video.precompute_montage --root /home/manya/argoverse --out docs/video/montage.pkl --scan 39472 --keep 15` | **~35–45 min** |
| 1b. quick scan (dev) | same, with `--scan 3000` | ~4 min |
| 1c. re-select only | same, with `--reuse_scores docs/video/montage.pkl --out docs/video/montage2.pkl` | ~1 min |
| 2. render | `python -u -m KD.video.render_montage --data docs/video/montage.pkl --outdir docs/video/frames` | ~1 min |
| 3. encode | the `ffmpeg` line step 2 prints | seconds |

Step 1 prints a progress line every 200 scenes:
`[N/39472] cov@p90  noKD=0.903  v1=0.711  v2=0.909`.

**The gate.** At the end of step 1 it prints a block headed `VALIDATION GATE`. To find it
in a log: `grep -A8 "VALIDATION GATE" <logfile>`. Read the `cov@p90` (last) column:

| arm | must be | if it is not |
|---|---|---|
| noKD | 0.903 | wrong checkpoint — see [CHECKPOINTS.md](CHECKPOINTS.md) |
| v1 | **0.711** | 0.754 means the old wrong v1 checkpoint is still wired in |
| v2 | 0.909 | wrong checkpoint |
| teacher | 0.894 | wrong checkpoint |

If those four numbers are right, the frames are trustworthy. If not, **stop** — do not
render, fix the checkpoint first.

To run detached and keep the log:
```bash
nohup python -u -m KD.video.precompute_montage --root /home/manya/argoverse \
    --out docs/video/montage.pkl --scan 39472 --keep 15 > scan.log 2>&1 &
tail -f scan.log          # live progress
```

`ffmpeg` **is installed** in `hivt_new` (`/home/manya/miniconda3/envs/hivt_new/bin/ffmpeg`).
It is not on the base `PATH`, so `which ffmpeg` outside the env reports nothing.

### 3.4 The montage renderer (built)

`KD/video/precompute_montage.py` (slow, torch + map) → pickle →
`KD/video/render_montage.py` (fast, no torch). The split exists so the visual can be
iterated without reloading four models and `ArgoverseMap` each time.

```bash
conda activate hivt_new
python -m KD.video.precompute_montage --root /home/manya/argoverse \
    --out <scratch>/montage.pkl --scan 3000 --keep 18
python -m KD.video.render_montage --data <scratch>/montage.pkl \
    --outdir <scratch>/frames            # then run the ffmpeg line it prints
```

The teacher is computed and stored but **not rendered** — a 4th panel can be added
later without re-running the scan.

**Validation gate (Phase A).** Aggregate coverage must reproduce the paper. At
n=60 scenes: noKD 0.906 / v1 0.746 / v2 0.899 / teacher 0.902, against the paper's
0.903 / 0.711 / 0.909 / 0.894. Confirms the renderer measures the same quantity
`eval.py` does.

**Four bugs found and fixed while building it** — all four were silent:

1. **`forward()` mutates `data.y` in place** ([models/hivt.py:122](../models/hivt.py#L122),
   `data.y = bmm(data.y, rotate_mat)`). Running three models on one batch rotates the
   ground truth three times; every arm after the first is scored against a wrong
   target and nothing errors. **Each model gets a fresh `.clone()`.**
2. **Lanes were queried around the AV, not the focal agent.** The scene frame is
   AV-anchored but the focal agent can be 100 m+ away — scene 1801 had lanes spanning
   `x[-88,105]` and the agent at `x[125,152]`. The map never covered the trajectory.
3. **Ranking by teacher mode-weight entropy selects visually trivial scenes.** Entropy
   saturates near ln6=1.792 on a straight road where all six modes overlap. It picked
   scenes with 2.2–2.5 m endpoint spread — *below* the teacher's 5.15 m average.
   Now ranked by **mean pairwise mode-endpoint distance** (the paper's own diversity
   quantity); picks land at 9–16 m. Both scores are stored for every scanned scene so
   the criterion can change without re-scanning.
4. **Framing included the full 2 s history**, which is about as long as the 3 s future,
   halving the scale of the part that matters and pushing the 2–5 m bands to sub-pixel.
   Now frames the future + last 1 s of history, with `aspect='equal',
   adjustable='datalim'` so panels fill without distortion and stay comparable.

**Scene selection is a percentile WINDOW, not the top-k.** Ranking 39,472 scenes by
mode spread and taking the top 18 selects the ~99.95th percentile — the scenes where
the model has no idea at all. Three things blow up together there: predicted scale
(bands of 17×20 m over a 6 m trajectory), best-of-6 error (v2 missing by 12 m), and
therefore the v1-vs-v2 contrast itself, because neither band covers a 12 m error. On
the rank-1 scene of that run the counts were **noKD 54/60, v1 34/60, v2 39/60** — the
from-scratch baseline won and the frame argued *against* the paper.

Current defaults: `--pct_lo 80 --pct_hi 97 --max_fde 3.0`, which yields ~11.8 m spread
and best-of-6 FDE of 0.1–2.6 m. Representative frame: **noKD 60/60, v1 37/60,
v2 52/60**. Ordinary tail-selection pathology — the extreme of any quality statistic is
dominated by outliers, and in forecasting the outliers are where the model is broken.

`--reuse_scores <pkl>` re-selects from an existing pickle's stored scores, skipping the
35-minute Phase A entirely.

**Open judgment call.** Per-scene counts fluctuate — on some scenes v1 scores 60/60
and beats v2. That is the 0.711 coverage being a *statistic*, and it is exactly why a
single scene cannot carry the shot. Do **not** reorder scenes to open on a
v1-unfavourable frame; that is cherry-picking by another name. Let the running counter
converge, and consider a caption making the point explicitly ("no single scene settles
this — watch the running total").

### 3.5 Known bugs in the existing visualizer

[visualisation_other_tests/hivt_visualize.py](../visualisation_other_tests/hivt_visualize.py):

1. **`--show_uncertainty` is dead code.** Line 186 does `pred = pred[..., :2]`, dropping the
   scale channels; the guard at line 287 (`pred_abs.shape[-1] >= 4`) is therefore *always
   false* and the ribbon never draws. Nobody has ever seen this feature work.
2. **The ribbon would be wrong if it did draw.** `fill_between(mu_x, mu_y ± b_y)` is a y-only
   band in the *scene* frame, but `b` is defined in the *agent* frame — wrong under any
   rotation — and `b_x` is ignored entirely.
3. `fill_between` needs monotonic x; a turning trajectory self-intersects.

The montage renderer should be a **new script**, not a patch of this one (that script's job is
per-mode diagnostic panels, a different purpose).

### 3.6 Environment

```
source /home/manya/miniconda3/etc/profile.d/conda.sh && conda activate hivt_new
```
Dataset root: `/home/manya/argoverse`. Gotcha: multiple `trainer.validate()` in one process
needs `num_workers=0` (fork-after-CUDA); see [handoff_kd_thesis_state.md](handoff_kd_thesis_state.md) §5.

---

## 4. Paper improvements, ranked by return

1. **Embedded latency on a real target** (Jetson Orin Nano / ARM SBC). Converts "55× fewer
   parameters" into a robotics claim, answers the venue-fit objection, *and* yields physical
   footage for the video — a board on a desk with a latency counter is the "visual demo"
   without a robot. [KD/profile_efficiency.py](../KD/profile_efficiency.py) exists; mostly a
   porting job.
2. **The matched HiVT-32 seed sweep.** The limitations section already concedes HiVT-32 is a
   single run while HiVT-16 has three seeds. Pure compute, no new code. Reviewers read the
   concession as "they knew and didn't do it."
3. **A trained Hungarian-matched KD baseline.** Currently argued against in Sec. IV but never
   benchmarked. The central claim — permutation-invariance matters — rests on a diagnostic
   (0% identity-optimal), not on a beaten baseline. **Strongest scientific gap.**
4. **Planner in the loop**, even a toy risk-aware IDM/braking rule showing v1's narrow intervals
   produce later braking than v2's. Most work; legitimate to leave as future work, but it is
   what would make this unambiguously a robotics paper.

---

## 5. Open question

The conversation was framed around **ICRA 2026**, which was held in June 2026. As of
2026-09-02 the open submission cycle is the following one (ICRA deadlines are typically
mid-September). **Confirm the target venue and deadline** — none of the above changes either
way, but the schedule does.

---

## 6. The video — BUILT

`docs/video/icra_kd.mp4` — 1280×720, **178.24 s**, **2.43 MB**, H.264/yuv420p.
Ten shots, captions burned in, anonymous (no authors, no affiliation, no repo URL).

### 6.1 Rebuild

```bash
source /home/manya/miniconda3/etc/profile.d/conda.sh && conda activate hivt_new
python -u -m KD.video.build_video                 # ~1 min, renders + encodes
python -u -m KD.video.build_video --shots 3,6     # iterate on a subset
python -u -m KD.video.build_video --no_encode     # frames only
```

No torch, no dataset, no `ArgoverseMap` — everything is read from precomputed
artefacts, so the visuals can be iterated freely.

| File | Role |
|---|---|
| `KD/video/vstyle.py` | palette, caption band, canvas, `FrameWriter` |
| `KD/video/shots.py` | the ten shot builders |
| `KD/video/build_video.py` | timeline (single source of truth), gates, ffmpeg |
| `docs/video/perm_trace500.npz` | Beat 1 data (new, see 6.3) |
| `docs/video/montage_illustrative.pkl` | Beat 2 data (existing, = `montage3.mp4`) |
| `docs/figures/fulldata_reliability.json` | reliability curves |

### 6.2 The timeline

| # | Time | Shot |
|---|---|---|
| 1 | 0:00 | title (anonymous) |
| 2 | 0:10 | why distillation: capacity cost, then classifier-vs-mixture |
| 3 | 0:30 | **BEAT 1** — WTA → the pairing + cost matrix → the matrix over 500 scenes |
| 4 | 1:03 | why not Hungarian; score under the whole mixture |
| 5 | 1:21 | v1's scale pathology; "best-of-K cannot see this" |
| 6 | 1:33 | **BEAT 2** — the 16-scene coverage montage → 0.903 / 0.711 / 0.909 |
| 7 | 1:59 | reliability curves |
| 8 | 2:11 | Table II → HiVT-16 seeds → frontier |
| 9 | 2:41 | three takeaways + the transfer claim |
| 10 | 2:55 | anonymous title reprise |

### 6.3 Beat 1's data — and why the diagnostic was re-run

`kd_mode_permutation_test.py` gained one additive flag, `--dump_per_scene`, which
saves the per-scene trace the loop already computes and then discards (cost
matrix, optimal permutation, greedy nearest, mode geometry, ground truth). Shot
3c *animates* the assignment matrix accumulating; without the trace the
accumulation order would have to be invented, which is fabricating a time series
whose endpoint happens to be real.

```bash
python -u -m KD.kd_mode_permutation_test \
    --teacher_ckpt checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt \
    --student_ckpt HiVT-32/gxhl2ug9/checkpoints/epoch=63-step=411903.ckpt \
    --root /home/manya/argoverse --num_scenes 500 \
    --out_dir docs/video --dump_per_scene docs/video/perm_trace500.npz
```

**The student checkpoint is `HiVT-32/gxhl2ug9/…epoch=63`.** Nothing recorded which
student produced the paper's Fig. 2, so it was identified by fingerprinting
against the existing 20-scene run: `greedy_freq` and `perm_freq` came back
bit-identical, all rates identical, differences only in the 7th significant
figure (float32 GPU noise). The 500-scene rerun likewise reproduces the paper
exactly — identical integer matrices, `identity_is_optimal_rate = 0.000`,
5.165 m / 1.817 m / 3.98×. The script asserts the trace replays to the aggregate.

Note this is *not* a CHECKPOINTS.md arm, and should not be: the diagnostic needs
only *an* independently trained student, and matching the published figure beats
matching the Table II arm here.

**New fact worth adding to the paper.** The same non-identity permutation
`[0 4 5 2 1 3]` is optimal in **84.4 %** of scenes, only **24 of 720**
permutations ever appear, and identity appears **zero** times. The mismatch is
systematic, not noise — a stronger statement than "0 % identity-optimal" alone.

### 6.4 Traps hit while building (all silent)

1. **The concat demuxer inflates variable `duration` directives.** A 179.98 s
   timeline encoded to **184 s** — over the hard cap, with nothing failing.
   Durations are now quantised to whole 1/25 s ticks emitted one per frame, and
   `build_video.py` re-probes the *encoded* duration rather than trusting the
   manifest.
2. **Captions ran off the right edge**, silently deleting the claim the shot
   existed to make. `caption()` now wraps, and `FrameWriter` checks every text
   bbox against the frame and reports overflows at render time.
3. **Framing on the trajectory, not the fan** — handoff bug #4 in a new place.
   The mode fan is ~7 m at the end of a ~30 m trajectory, so a
   trajectory-sized window renders it as a smudge. `_modes_axes(focus='ends')`
   crops to the endpoints and sizes the window to the *axes box aspect*;
   forcing a square window wastes most of a 16:9 frame.
4. **HiVT-32's modes fan longitudinally, not angularly.** Largest angular spread
   between mode endpoints over 500 scenes is only ~2°: the modes differ in how
   far the agent travels, not which way it turns. Selecting scenes for angular
   diversity is therefore pointless — the pairing argument is carried by the
   6×6 cost matrix panel instead.
5. **noKD and v2 coincide on the reliability plot**, so equal-weight lines left
   the blue invisible against a three-item legend. The baseline is now a wide
   soft band under a thin v2 line — both visible, and the coincidence is the
   thing you notice.

### 6.5 Gates enforced on every build

- montage cov@p90 must be 0.903 / 0.711 / 0.909 / 0.894 → else refuse to render
- encoded duration ≤ 180 s (currently 178.24, margin 1.76 s)
- file size ≤ 25 MB (currently 2.43 MB, margin 23.8 MB)
- zero text overflows
