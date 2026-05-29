"""
prepare_teacher_cache.py

Convert a directory of per-scene teacher .pt files into a single indexed HDF5
cache for faster random access during student KD training.

Example:
    python prepare_teacher_cache.py \
        --input_dir teacher_outputs/train \
        --output_h5 teacher_outputs/train.h5
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import torch
from tqdm import tqdm


def _compression_kwargs(name: str, level: int):
    if name == "none":
        return {}
    if name == "lzf":
        return {"compression": "lzf"}
    return {"compression": "gzip", "compression_opts": level}


def _normalize_seq_id(seq_id) -> str:
    if isinstance(seq_id, torch.Tensor):
        if seq_id.numel() == 1:
            seq_id = seq_id.item()
        else:
            seq_id = seq_id.tolist()
    if isinstance(seq_id, (list, tuple)) and len(seq_id) == 1:
        seq_id = seq_id[0]
    return str(seq_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with per-scene .pt files")
    parser.add_argument("--output_h5", required=True, help="Path to the output .h5 cache")
    parser.add_argument("--compression", choices=["none", "gzip", "lzf"], default="gzip")
    parser.add_argument("--compression_level", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--delete_source",
        action="store_true",
        help="Delete each source .pt file after it has been written to the HDF5 cache",
    )
    parser.add_argument("--manifest", type=str, default=None,
                        help="Optional manifest.json to copy into the HDF5 metadata")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_h5 = Path(args.output_h5)

    if output_h5.exists() and not args.overwrite:
        raise FileExistsError(f"{output_h5} already exists. Use --overwrite to replace it.")

    teacher_files = sorted(input_dir.glob("*.pt"))
    if not teacher_files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}")

    compression_kwargs = _compression_kwargs(args.compression, args.compression_level)

    manifest_data = None
    manifest_path = Path(args.manifest) if args.manifest else input_dir / "manifest.json"
    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text())

    with h5py.File(output_h5, "w") as h5_file:
        meta = h5_file.create_group("_meta")
        meta.attrs["format"] = "hivt_teacher_cache_v1"
        meta.attrs["source_dir"] = str(input_dir.resolve())
        meta.attrs["num_files"] = len(teacher_files)
        meta.attrs["compression"] = args.compression
        if manifest_data is not None:
            meta.attrs["teacher_params_total"] = int(manifest_data["teacher_params"]["total"])
            meta.attrs["teacher_params_trainable"] = int(manifest_data["teacher_params"]["trainable"])

        for teacher_file in tqdm(teacher_files, desc="Writing teacher cache"):
            teacher = torch.load(teacher_file, map_location="cpu")
            seq_id = _normalize_seq_id(teacher.get("seq_id", teacher_file.stem))

            if seq_id in h5_file:
                del h5_file[seq_id]

            group = h5_file.create_group(seq_id)
            group.attrs["seq_id"] = seq_id
            group.create_dataset("loc", data=teacher["loc"].cpu().numpy(), **compression_kwargs)
            group.create_dataset("scale", data=teacher["scale"].cpu().numpy(), **compression_kwargs)
            group.create_dataset("pi", data=teacher["pi"].cpu().numpy(), **compression_kwargs)

            if args.delete_source:
                os.remove(teacher_file)

    print(f"Wrote {len(teacher_files)} teacher records to {output_h5}")


if __name__ == "__main__":
    main()
