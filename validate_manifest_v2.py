#!/usr/bin/env python3
"""List unfilled (null) fields of a run manifest.

  validate_manifest_v2.py configs/run_manifest_v2.yaml            # report
  validate_manifest_v2.py configs/run_manifest_v2.yaml --strict   # exit 1 if any null
"""
import argparse, sys
import yaml


def walk(d, prefix=""):
    for k, v in d.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from walk(v, p + ".")
        else:
            yield p, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    flat = dict(walk(yaml.safe_load(open(a.manifest))))
    missing = [k for k, v in flat.items() if v is None]
    print(f"{len(flat) - len(missing)}/{len(flat)} fields filled")
    for k in missing:
        print("  missing:", k)
    sys.exit(1 if (a.strict and missing) else 0)


if __name__ == "__main__":
    main()
