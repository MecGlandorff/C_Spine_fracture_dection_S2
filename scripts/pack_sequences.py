import os
import glob
import numpy as np
from collections import defaultdict

# Input: folder met {uid}_{c}_{t}.npy
SRC_DIR = r"d:/AI DATA/NEXUS/data/masks/cropped2d"

# Output: folder met {uid}_{c}.npy (T,H,W,6)
DST_DIR = r"d:/AI DATA/NEXUS/data/masks/cropped_packed"

T_EXPECT = 15

os.makedirs(DST_DIR, exist_ok=True)

files = glob.glob(os.path.join(SRC_DIR, "*.npy"))
if len(files) == 0:
    raise RuntimeError(f"No npy files found in SRC_DIR:{SRC_DIR}")

groups = defaultdict(dict)

for fp in files:
    name = os.path.basename(fp).replace(".npy", "")

    # name looks like: <uid>_<c>_<t>
    # uid contains dots, so split from the right side
    uid_c, t = name.rsplit("_", 1)
    uid, c = uid_c.rsplit("_", 1)

    groups[(uid, int(c))][int(t)] = fp

print(f"Found {len(groups)} vertebra groups")

written = 0
skipped = 0

for (uid, c), slices in groups.items():
    if len(slices) != T_EXPECT:
        skipped += 1
        continue

    seq = []
    for t in range(T_EXPECT):
        fp = slices.get(t, None)
        if fp is None:
            raise RuntimeError(f"Missing slice t={t} for {uid} C{c}")
        arr = np.load(fp)

        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            raise ValueError(f"Expected (H,W,6), got {getattr(arr,'shape',None)} in {fp}")
        if arr.shape[-1] != 6:
            raise ValueError(f"Expected 6 channels, got {arr.shape} in {fp}")

        seq.append(arr)

    seq = np.stack(seq, axis=0)  # (T,H,W,6)

    out_fp = os.path.join(DST_DIR, f"{uid}_{c}.npy")
    np.save(out_fp, seq)
    written += 1

print(f"Done. Written: {written}, skipped (not exactly {T_EXPECT} slices): {skipped}")
print(f"Packed folder: {DST_DIR}")
