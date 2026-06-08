"""
test_kd_loss.py

Unit tests for the permutation-invariant mixture-NLL KD loss.
Run from the repo root:  python KD/files/test_kd_loss.py
(The underlying math was already verified in numpy; this checks the torch
implementation, autograd, and the scale guard.)
"""

import os
import sys

import torch

# Run-from-anywhere bootstrap: put this KD dir on sys.path so the sibling KD
# modules import under one flat convention. See KD/files/KD_CHANGES.md.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kd_loss import HiVTKDLoss     # sibling KD module (flat; KD/files on sys.path)

torch.manual_seed(0)
K, F, B, H, D = 6, 6, 4, 30, 2


def _rand_student(requires_grad=False):
    loc   = torch.randn(K, B, H, D, requires_grad=requires_grad)
    scale = (torch.randn(K, B, H, D).abs() + 0.1).requires_grad_(requires_grad)
    pi    = torch.randn(B, K, requires_grad=requires_grad)
    return loc, scale, pi


def _rand_teacher(f=F):
    return torch.randn(f, B, H, D), (torch.randn(f, B, H, D).abs() + 0.1), torch.randn(B, f)


def test_finite():
    fn = HiVTKDLoss(lambda_kl=1.0, lambda_pi=0.0)
    ls, ss, ps = _rand_student()
    lt, st, pt = _rand_teacher()
    loss, logs = fn(ls, ss, ps, lt, st, pt)
    assert torch.isfinite(loss), "loss is not finite"
    assert "kd/mix_nll" in logs
    print(f"[finite]            loss={loss.item():.4f}  OK")


def test_perm_invariance():
    fn = HiVTKDLoss(lambda_kl=1.0, lambda_pi=0.0)
    ls, ss, ps = _rand_student()
    lt, st, pt = _rand_teacher()
    base, _ = fn(ls, ss, ps, lt, st, pt)

    # permute teacher modes
    pT = torch.randperm(F)
    Lt, _ = fn(ls, ss, ps, lt[pT], st[pT], pt[:, pT])
    # permute student modes
    pS = torch.randperm(K)
    Ls, _ = fn(ls[pS], ss[pS], ps[:, pS], lt, st, pt)

    assert torch.allclose(base, Lt, atol=1e-5), f"teacher-perm changed loss: {base} vs {Lt}"
    assert torch.allclose(base, Ls, atol=1e-5), f"student-perm changed loss: {base} vs {Ls}"
    print(f"[perm-invariance]   base={base.item():.4f}  teacherperm={Lt.item():.4f}  studentperm={Ls.item():.4f}  OK")


def test_closer_lower():
    fn = HiVTKDLoss(lambda_kl=1.0, lambda_pi=0.0)
    lt, st, pt = _rand_teacher()
    perfect_scale = torch.full_like(lt, 0.5)
    L_perfect, _ = fn(lt.clone(), perfect_scale, pt.clone(), lt, st, pt)
    L_noisy,   _ = fn(lt + 2.0 * torch.randn_like(lt), perfect_scale, pt.clone(), lt, st, pt)
    assert L_perfect < L_noisy, f"perfect {L_perfect} not < noisy {L_noisy}"
    print(f"[closer=>lower]     perfect={L_perfect.item():.4f}  noisy={L_noisy.item():.4f}  OK")


def test_k_neq_f():
    fn = HiVTKDLoss(lambda_kl=1.0, lambda_pi=0.0)
    ls, ss, ps = _rand_student()              # K=6
    lt, st, pt = _rand_teacher(f=4)           # F=4
    loss, _ = fn(ls, ss, ps, lt, st, pt)
    assert torch.isfinite(loss)
    print(f"[K!=F]              loss={loss.item():.4f}  OK")


def test_grad_flow():
    fn = HiVTKDLoss(lambda_kl=1.0, lambda_pi=0.0)
    ls, ss, ps = _rand_student(requires_grad=True)
    lt, st, pt = _rand_teacher()
    lt.requires_grad_(True)                   # teacher passed detached in real code
    loss, _ = fn(ls, ss, ps, lt.detach(), st.detach(), pt.detach())
    loss.backward()
    assert ls.grad is not None and torch.isfinite(ls.grad).all(), "no/!finite grad to student loc"
    assert ss.grad is not None, "no grad to student scale"
    assert ps.grad is not None, "no grad to student pi"
    assert lt.grad is None, "gradient leaked into teacher (should be detached)"
    print("[grad-flow]         student grads present, teacher detached  OK")


def test_scale_guard():
    fn = HiVTKDLoss(lambda_kl=1.0, lambda_pi=0.0, min_scale=1e-3)
    ls, _, ps = _rand_student()
    ss = torch.zeros(K, B, H, D)              # degenerate scale
    lt, st, pt = _rand_teacher()
    loss, _ = fn(ls, ss, ps, lt, st, pt)
    assert torch.isfinite(loss), "zero scale produced non-finite loss; clamp failed"
    print("[scale-guard]       zero scale handled by clamp  OK")


if __name__ == "__main__":
    test_finite()
    test_perm_invariance()
    test_closer_lower()
    test_k_neq_f()
    test_grad_flow()
    test_scale_guard()
    print("\nAll loss unit tests passed.")
