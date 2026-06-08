"""
smoke_test_pipeline.py

Run BEFORE launching KD training. Five staged checks, each prints PASS/FAIL and
the reason. Stops being useful only if a stage errors out — read the last
printed stage to see where it broke.

What each stage proves:
  0. The H5 cache opens and its keys are in the format the loader expects.
  1. A teacher record loads with correct shapes, no NaNs, positive scale.
  2. KEY FORMAT: the seq_id coming off the real dataset resolves to a cache
     entry (this is the check that catches the silent has_teacher=False bug).
  3. KDDataset attaches teacher tensors -> has_teacher is True for real items.
  4. PyG batching stacks teacher tensors to [B,F,H,2] / [B,F] and the
     train-time permute to [F,B,H,2] is consistent with num_graphs.

Run from the repo root (flat import convention; see KD/files/KD_CHANGES.md).

Usage (HDF5 cache):
python -m KD.smoke_test_pipeline --root /home/manya/argoverse --teacher /home/manya/HiVT/teacher_outputs/train.h5 --n 64
why --n 64? It's a reasonable sample size to check that has_teacher=True items are present without taking too long. Adjust as needed for larger/smaller datasets or if you want more/less confidence in the coverage check.
Usage (directory of .pt files):
    python -m KD.smoke_test_pipeline --root ... --teacher teacher_outputs/train --n 64
"""

import argparse
import os
import sys

import torch

# Run-from-anywhere bootstrap: put the repo root and this KD dir on sys.path so
# the repo packages (datasets) and the sibling KD modules import under one flat
# convention. See KD/files/KD_CHANGES.md.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets import ArgoverseV1Dataset                 # repo module (absolute)
from kd_teacher_store_fixed import TeacherStore          # sibling KD module (flat)
from kd_dataset import KDDataset, KDData                 # sibling KD module (flat)


def _ok(msg):   print(f"  PASS  {msg}")
def _bad(msg):  print(f"  FAIL  {msg}")


def stage0(store):
    print("\n[0] H5 / store opens and key format is inspectable")
    if getattr(store, "_is_h5", False):
        keys = [k for k in store._h5.keys() if k != "_meta"]
        print(f"      total keys: {len(keys)}")
        print(f"      sample keys: {keys[:5]}")
        if "_meta" in store._h5:
            print(f"      _meta attrs: {dict(store._h5['_meta'].attrs)}")
        looks_tensor = any(k.startswith("tensor(") for k in keys[:50])
        print(f"      keys look like 'tensor(<id>)': {looks_tensor}")
        _ok("store opened")
        return keys[:1][0] if keys else None
    else:
        files = list(store.path.glob("*.pt"))
        print(f"      total .pt files: {len(files)}")
        print(f"      sample stems: {[f.stem for f in files[:5]]}")
        _ok("directory store opened")
        return files[0].stem if files else None


def stage1(store, sample_key):
    print("\n[1] One teacher record loads with correct shape / no NaN / scale>0")
    if sample_key is None:
        _bad("no records in cache"); return False
    rec = store.load(sample_key)
    if rec is None:
        _bad(f"store.load('{sample_key}') returned None"); return False
    ok = True
    for k, exp in (("loc", (6, 30, 2)), ("scale", (6, 30, 2)), ("pi", (6,))):
        got = tuple(rec[k].shape)
        if got != exp:
            _bad(f"{k} shape {got} != {exp}"); ok = False
    if torch.isnan(rec["loc"]).any() or torch.isnan(rec["scale"]).any() or torch.isnan(rec["pi"]).any():
        _bad("NaNs present"); ok = False
    smin = rec["scale"].min().item()
    if smin <= 0:
        _bad(f"scale min {smin} <= 0 (mixture NLL needs positive scale)"); ok = False
    else:
        print(f"      scale min/mean/max: {smin:.4f} / {rec['scale'].mean():.4f} / {rec['scale'].max():.4f}")
    if ok: _ok("record healthy")
    return ok


def stage2(store, base):
    print("\n[2] KEY FORMAT: real dataset seq_id resolves to a cache entry")
    data0 = base[0]
    seq_id = data0.seq_id
    plain  = KDDataset._normalize_seq_id(seq_id)
    tensorfmt = f"tensor({plain})"
    e_plain  = store.exists(plain)
    e_tensor = store.exists(tensorfmt)
    print(f"      raw seq_id        : {seq_id!r} (type {type(seq_id).__name__})")
    print(f"      normalized key    : '{plain}'        exists in cache: {e_plain}")
    print(f"      legacy tensor key : '{tensorfmt}'    exists in cache: {e_tensor}")
    if e_plain or e_tensor:
        _ok(f"seq_id resolves via {'plain' if e_plain else 'tensor()'} key "
            f"(KDDataset handles both)")
        return True
    _bad("seq_id resolves to NEITHER key form -> KD would be silently skipped. "
         "Check that the cache was built from THIS dataset split.")
    return False


def stage3(base, teacher, n):
    print(f"\n[3] KDDataset attaches teacher tensors (checking {n} items)")
    kd = KDDataset(base, teacher_dir=teacher)
    hits = 0
    for i in range(min(n, len(kd))):
        d = kd[i]
        ht = bool(d.has_teacher.all().item()) if torch.is_tensor(d.has_teacher) else bool(d.has_teacher)
        if ht and hasattr(d, "teacher_loc"):
            hits += 1
    print(f"      has_teacher=True for {hits}/{min(n, len(kd))} items")
    if hits == 0:
        _bad("zero items got teacher targets -> KD is OFF. (Likely key mismatch.)")
        return None
    if hits < min(n, len(kd)):
        print("      NOTE: partial coverage; KD will run only on covered scenes.")
    _ok("teacher targets attached")
    return kd


def stage4(kd):
    print("\n[4] PyG batching stacks teacher tensors correctly")
    from torch_geometric.data import DataLoader as PyGDataLoader
    loader = PyGDataLoader(kd, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    B = batch.num_graphs
    ok = True
    for k, want in (("teacher_loc", (B, 6, 30, 2)),
                    ("teacher_scale", (B, 6, 30, 2)),
                    ("teacher_pi", (B, 6))):
        if not hasattr(batch, k):
            _bad(f"batch missing {k}"); ok = False; continue
        got = tuple(getattr(batch, k).shape)
        if got != want:
            _bad(f"{k} batched shape {got} != {want}"); ok = False
        else:
            print(f"      {k}: {got}")
    # the permute used in training_step
    loc_t = batch.teacher_loc.permute(1, 0, 2, 3)   # -> [F,B,H,2]
    if tuple(loc_t.shape) != (6, B, 30, 2):
        _bad(f"permuted teacher_loc {tuple(loc_t.shape)} != (6,{B},30,2)"); ok = False
    if ok: _ok("batching + train-time permute consistent")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Argoverse dataset root")
    ap.add_argument("--teacher", required=True, help="teacher .h5 file OR dir of .pt files")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=64, help="items to probe for has_teacher")
    args = ap.parse_args()

    print("=" * 64)
    print("KD PIPELINE SMOKE TEST")
    print("=" * 64)

    store = TeacherStore(args.teacher, cache_size=0)
    sample_key = stage0(store)
    s1 = stage1(store, sample_key)

    print("\nBuilding base dataset (first run may preprocess; be patient) ...")
    base = ArgoverseV1Dataset(root=args.root, split=args.split, local_radius=50)

    s2 = stage2(store, base)
    kd = stage3(base, args.teacher, args.n)
    s4 = stage4(kd) if kd is not None else False

    print("\n" + "=" * 64)
    results = {"stage1_record": s1, "stage2_keyformat": s2,
               "stage3_wiring": kd is not None, "stage4_batching": s4}
    for k, v in results.items():
        print(f"  {k:20s}: {'PASS' if v else 'FAIL'}")
    all_ok = all(results.values())
    print("=" * 64)
    print("ALL CHECKS PASSED — safe to start training." if all_ok
          else "SOME CHECKS FAILED — fix before training (see stages above).")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
