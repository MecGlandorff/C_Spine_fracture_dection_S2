# scripts/train.py
# Program: Cervical fracture detection of C1-C7 (stage 2)
# Author: Mec Glandorff
# Version: 3.2-proto
# Description:
#   Training entrypoint.
#
# Usage:
#   python scripts/train.py --config configs/train.yaml

import os
import sys
import argparse
import yaml

# Allow "src" imports without installing the package. Temporary workaround during development. 
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from cspine_cls.train import run_train 


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found:{path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict in yaml")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to train.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # Crash the code early if missing stuff.
    for k in ["data_dir", "csv_path", "backbone"]:
        if k not in cfg or cfg[k] in [None, ""]:
            raise ValueError(f"Missing required config key: {k}")

    out_dir = run_train(cfg)
    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
