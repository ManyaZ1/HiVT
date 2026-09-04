"""
build_video.py — render the ICRA submission video end to end.

    conda activate hivt_new
    python -u -m KD.video.build_video

Renders every shot to a PNG sequence plus an ffmpeg concat manifest, then
encodes. No torch and no dataset access: everything is read from the
precomputed artefacts listed in shots.py, so a full rebuild is ~1 minute and
the visuals can be iterated on freely.

Hard constraints this script enforces, because both are disqualifying at
submission time rather than fixable later:

  * total duration <= 180 s  (ICRA hard cap)
  * output size    <= 25 MB  (PaperPlaza attachment limit)

Both are asserted, not hoped for. `--shots 3,6` renders a subset for quick
iteration; the duration check is skipped when a subset is selected.
"""

import argparse
import os
import shutil
import subprocess
import sys

from KD.video import shots as S
from KD.video.vstyle import FrameWriter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAP_SECONDS = 179.0   # 3:00 hard cap, with a second of margin
CAP_BYTES = 25 * 1024 * 1024

# The timeline. LIST ORDER is playback order -- the numbers are just shot ids
# (11 was added after 8 and keeps its id so --shots 11 still selects it).
# Budget is the design intent; actual is measured after render and reported per
# shot so drift is visible.
TIMELINE = [
    (1, 'title', 10.0),
    (2, 'why distillation', 20.0),
    (3, 'BEAT 1 — mode permutation', 14.0),
    (4, 'the objective', 18.0),
    (5, 'why the mean target breaks calibration', 12.0),
    (6, 'BEAT 2 — coverage montage', 	35.4),
    (7, 'reliability', 12.0),
    (8, 'results', 29.5),
    (11, 'deployment footprint', 9.0),
    (9, 'takeaway', 14.0),
    (10, 'reprise', 3.5),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--perm', default='docs/video/perm_trace500.npz')
    ap.add_argument('--montage', default='docs/video/montage_illustrative.pkl')
    ap.add_argument('--reliability', default='docs/figures/fulldata_reliability.json')
    ap.add_argument('--outdir', default='docs/video/final')
    ap.add_argument('--out', default='docs/video/icra_kd.mp4')
    ap.add_argument('--crf', type=int, default=15)
    ap.add_argument('--shots', default=None,
                    help='comma-separated subset, e.g. 3,6 (skips the duration check)')
    ap.add_argument('--no_encode', action='store_true')
    args = ap.parse_args()

    want = None if not args.shots else {int(x) for x in args.shots.split(',')}

    tr = S.load_perm(args.perm)
    mont = S.load_montage(args.montage)
    rel = S.load_reliability(args.reliability)
    eff = S.load_efficiency()

    # Refuse to build Beat 2 from a montage that does not reproduce the paper's
    # coverage -- a montage off the gate argues for the wrong conclusion.
    gate = {'noKD': 0.903, 'v1': 0.711, 'v2': 0.909, 'teacher': 0.894}
    li = list(mont['levels']).index(mont['band_level'])
    for arm, expect in gate.items():
        got = mont['totals'][arm][li] / mont['n_points']
        assert abs(got - expect) < 0.002, \
            f'montage gate FAILED: {arm} cov@p90={got:.4f}, expected {expect} ' \
            f'-- wrong checkpoint, see docs/CHECKPOINTS.md'
    print(f'montage gate OK  ' + '  '.join(
        f'{a}={mont["totals"][a][li] / mont["n_points"]:.3f}' for a in gate))

    builders = {
        1: lambda fw: S.shot01_title(fw),
        2: lambda fw: S.shot02_why_kd(fw),
        3: lambda fw: S.shot03_permutation(fw, tr),
        4: lambda fw: S.shot04_objective(fw, tr),
        5: lambda fw: S.shot05_pathology(fw),
        6: lambda fw: S.shot06_montage(fw, mont),
        7: lambda fw: S.shot07_reliability(fw, rel),
        8: lambda fw: S.shot08_results(fw),
        11: lambda fw: S.shot11_efficiency(fw, eff),
        9: lambda fw: S.shot09_takeaway(fw),
        10: lambda fw: S.shot10_reprise(fw),
    }

    if os.path.isdir(args.outdir):
        shutil.rmtree(args.outdir)
    fw = FrameWriter(args.outdir)

    print(f'\n{"shot":>4}  {"name":<40} {"budget":>7} {"actual":>7} {"frames":>7}')
    total_budget = 0.0
    for num, name, budget in TIMELINE:
        if want and num not in want:
            continue
        t0, n0 = fw.duration, fw.n
        builders[num](fw)
        total_budget += budget
        drift = fw.duration - t0 - budget
        flag = '' if abs(drift) < 0.05 else f'  ({drift:+.2f}s)'
        print(f'{num:>4}  {name:<40} {budget:>6.1f}s {fw.duration - t0:>6.1f}s '
              f'{fw.n - n0:>7}{flag}')

    print(f'\ntotal: {fw.duration:.2f} s in {fw.n} frames '
          f'(budget {total_budget:.1f} s)')

    if fw.overflows:
        print(f'\n{len(fw.overflows)} TEXT OVERFLOW(S) — these are cut off on screen:')
        for frame, text, bb in fw.overflows:
            print(f'  f{frame:05d}  {bb}  {text}')
    else:
        print('no text overflow')
    if not want:
        assert fw.duration <= CAP_SECONDS + 1e-6, \
            f'OVER THE ICRA CAP: {fw.duration:.2f}s > {CAP_SECONDS}s'

    manifest = fw.write_manifest()
    if args.no_encode:
        print(f'frames + manifest in {args.outdir} (encode skipped)')
        return

    out = os.path.abspath(args.out)
    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i',
           os.path.basename(manifest),
           '-vf', 'fps=25,format=yuv420p',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', str(args.crf),
           '-movflags', '+faststart', out]
    print('\n' + ' '.join(cmd))
    r = subprocess.run(cmd, cwd=args.outdir, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        sys.exit('ffmpeg failed')

    # Verify the ENCODED duration, not the manifest's intent. These differed by
    # 2.6 s once already (see FrameWriter), and a 3:01 video is rejected at
    # upload with no chance to fix it.
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', out], capture_output=True, text=True)
    encoded = float(probe.stdout.strip()) if probe.returncode == 0 else float('nan')

    size = os.path.getsize(out)
    print(f'\nwrote {out}\n  {size / 1e6:.2f} MB   {encoded:.2f} s encoded '
          f'(manifest {fw.duration:.2f} s)   {size * 8 / encoded / 1e3:.0f} kbps')
    if not want:
        assert size <= CAP_BYTES, \
            f'OVER THE PAPERPLAZA LIMIT: {size / 1e6:.1f} MB > 25 MB -- raise --crf'
        assert encoded <= 180.0, \
            f'ENCODED DURATION OVER THE ICRA CAP: {encoded:.2f}s > 180s'
        print(f'  OK: under the 180 s cap by {180.0 - encoded:.2f} s, '
              f'under the 25 MB limit by {(CAP_BYTES - size) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
