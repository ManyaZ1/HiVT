"""
One-off migration: rewrite an old teacher cache whose scene groups were keyed
as ``tensor(<seq_id>)`` (the pre-normalize_seq_id bug) into a fresh file keyed
by the plain ``<seq_id>`` string the datamodule looks up.

Group payloads (loc/scale/pi) and the ``_meta`` group are copied verbatim with
their original compression/chunking. The source file is left untouched.

    python KD/fix_teacher_cache_keys.py SRC.h5 DST.h5
"""
import re
import sys

import h5py

PAT = re.compile(r"^tensor\((\d+)\)$")


def main(src_path: str, dst_path: str) -> None:
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        keys = list(src.keys())
        total = len(keys)
        renamed = passthrough = skipped = 0
        for i, key in enumerate(keys):
            if key == "_meta":
                src.copy(src[key], dst, name="_meta")
                passthrough += 1
                continue
            m = PAT.match(key)
            new_key = m.group(1) if m else key
            if not m:
                # Already-clean or unexpected key: copy under its own name.
                passthrough += 1
            else:
                renamed += 1
            if new_key in dst:
                skipped += 1
                continue
            src.copy(src[key], dst, name=new_key)
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1}/{total} groups copied ...", flush=True)
        print(
            f"Done. {total} source groups -> renamed={renamed} "
            f"passthrough={passthrough} skipped_dupes={skipped}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python KD/fix_teacher_cache_keys.py SRC.h5 DST.h5")
    main(sys.argv[1], sys.argv[2])
