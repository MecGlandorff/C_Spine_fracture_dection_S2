# Program: Cervical fracture detection of C1-C7 (stage 2)
# Author: Mec Glandorff
# Version: 3.2-proto
# Description:
#   Inference of prototupe.
#
# Usage:
#   python scripts/infer.py --config configs/infer.yaml --uid <StudyInstanceUID> 

import os
import sys
import argparse
import json
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC) #!!!REMINDER FOR SELF Temporary sys path mod for developemnt. This has to be changed to be packaged and installed (python -e)
from cspine_cls.inference import score_patient 


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict in yaml.")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to infer.yaml")
    ap.add_argument("--uid", required=True, type=str, help="StudyInstanceUID")
    ap.add_argument("--out", default=None, type=str, help="Optional path to save json")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    for k in ["data_dir", "checkpoint_path"]:
        if k not in cfg or cfg[k] in [None, ""]:
            raise ValueError(f"Missing required config key: {k}")

    out = score_patient(args.uid, cfg["checkpoint_path"], cfg)

    print(json.dumps(out, indent=2))

    if args.out is not None:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
