import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np


def _parse_unpacked_name(path: str) -> Tuple[str, int, int]:
    """Parse <uid>_<c>_<t>.npy while allowing dots/underscores in uid."""
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        uid_c, t = name.rsplit("_", 1)
        uid, c = uid_c.rsplit("_", 1)
        return uid, int(c), int(t)
    except ValueError as exc:
        raise ValueError(f"Expected filename '<uid>_<c>_<t>.npy', got: {path}") from exc


def pack_sequences(
    src_dir: str,
    dst_dir: str,
    n_slices: int = 15,
    overwrite: bool = False,
    strict: bool = False,
) -> Dict[str, int]:
    os.makedirs(dst_dir, exist_ok=True)

    files = glob.glob(os.path.join(src_dir, "*.npy"))
    if len(files) == 0:
        raise RuntimeError(f"No npy files found in src_dir: {src_dir}")

    groups = defaultdict(dict)
    for fp in files:
        uid, c, t = _parse_unpacked_name(fp)
        groups[(uid, c)][t] = fp

    written = 0
    skipped_incomplete = 0
    skipped_existing = 0

    for (uid, c), slices in sorted(groups.items()):
        missing = [t for t in range(n_slices) if t not in slices]
        if missing:
            if strict:
                raise RuntimeError(f"Missing slices {missing} for {uid} C{c}")
            skipped_incomplete += 1
            continue

        out_fp = os.path.join(dst_dir, f"{uid}_{c}.npy")
        if os.path.exists(out_fp) and not overwrite:
            skipped_existing += 1
            continue

        seq = []
        for t in range(n_slices):
            fp = slices[t]
            arr = np.load(fp, allow_pickle=False)

            if not isinstance(arr, np.ndarray) or arr.ndim != 3:
                raise ValueError(f"Expected (H,W,6), got {getattr(arr, 'shape', None)} in {fp}")
            if arr.shape[-1] != 6:
                raise ValueError(f"Expected 6 channels, got {arr.shape} in {fp}")

            seq.append(arr)

        np.save(out_fp, np.stack(seq, axis=0))
        written += 1

    return {
        "groups": len(groups),
        "written": written,
        "skipped_incomplete": skipped_incomplete,
        "skipped_existing": skipped_existing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack unpacked vertebra timestep files into one npy per vertebra."
    )
    parser.add_argument("--src-dir", required=True, help="Folder with <uid>_<c>_<t>.npy files")
    parser.add_argument("--dst-dir", required=True, help="Output folder for <uid>_<c>.npy files")
    parser.add_argument("--n-slices", type=int, default=15, help="Expected timesteps per vertebra")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing packed files")
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping incomplete groups")
    args = parser.parse_args()

    stats = pack_sequences(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        n_slices=args.n_slices,
        overwrite=args.overwrite,
        strict=args.strict,
    )

    print(f"Found vertebra groups: {stats['groups']}")
    print(f"Written: {stats['written']}")
    print(f"Skipped incomplete: {stats['skipped_incomplete']}")
    print(f"Skipped existing: {stats['skipped_existing']}")
    print(f"Packed folder: {args.dst_dir}")


if __name__ == "__main__":
    main()
