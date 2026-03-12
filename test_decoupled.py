#!/usr/bin/env python3
"""
Run Multi-Compartment Segmentation (Girder-free) and save JSON outputs locally.

Usage inside the Docker container
-----------------------------------
    python /opt/MultiC/test_decoupled.py \\
        --input_file  /input/slide.svs \\
        --modelfile   /model/FUSION_MCS_FFPE_v1.pth \\
        --output_dir  /output/

Mount your local directories with -v when calling docker run (see bottom of this
file for the full command).

Optional tuning flags (all have defaults matching the Girder plugin)
----------------------------------------------------------------------
    --boxSize              INT    Tile size in pixels          (default: 2048)
    --bordercrop           INT    Border pixels to zero out    (default: 300)
    --roi_thresh           FLOAT  Detection score threshold    (default: 0.01)
    --white_percent        FLOAT  Min tissue fraction per tile (default: 0.01)
    --overlap_percentHR    FLOAT  Tile overlap 0-1            (default: 0)
    --Mag20X                      Flag: slide is 20X magnification
    --no_interstitium             Flag: suppress interstitium output
"""

import argparse
import json
import os
import sys

# -----------------------------------------------------------------------
# Make the multic package importable when PYTHONPATH is not pre-set
# (e.g., running locally outside Docker).
# -----------------------------------------------------------------------
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-Compartment Segmentation — Girder-free runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input_file",  required=True,
                   help="Path to the whole slide image (.svs / .tif / .scn)")
    p.add_argument("--modelfile",   required=True,
                   help="Path to the Detectron2 weights (.pth)")
    p.add_argument("--output_dir",  required=True,
                   help="Directory where per-class JSON files will be saved")

    # Tuning — all default to the same values as the Girder plugin
    p.add_argument("--boxSize",           type=int,   default=2048)
    p.add_argument("--bordercrop",        type=int,   default=300)
    p.add_argument("--roi_thresh",        type=float, default=0.01)
    p.add_argument("--white_percent",     type=float, default=0.01)
    p.add_argument("--overlap_percentHR", type=float, default=0)
    p.add_argument("--downsampleRateHR",  type=int,   default=1)
    p.add_argument("--chop_thumbnail_resolution", type=int, default=16)
    p.add_argument("--Mag20X",            action="store_true", default=False)
    p.add_argument("--no_interstitium",   action="store_true", default=False)
    p.add_argument("--min_size",          type=int,   nargs=6,
                   default=[30, 30, 30, 30, 30, 30],
                   metavar="N",
                   help="Min contour area for each of the 6 classes")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if not os.path.isfile(args.input_file):
        sys.exit(f"ERROR: Slide not found: {args.input_file}")
    if not os.path.isfile(args.modelfile):
        sys.exit(f"ERROR: Model not found: {args.modelfile}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Convert no_interstitium flag -> show_interstitium bool
    args.show_interstitium = not args.no_interstitium

    # predict() reads args.file for the slide path
    args.file = args.input_file

    print("=" * 60)
    print("Multi-Compartment Segmentation — Decoupled Run")
    print("=" * 60)
    print(f"  Slide  : {args.input_file}")
    print(f"  Model  : {args.modelfile}")
    print(f"  Output : {args.output_dir}")
    print(f"  boxSize: {args.boxSize}  bordercrop: {args.bordercrop}")
    print(f"  roi_thresh: {args.roi_thresh}  Mag20X: {args.Mag20X}")
    print("=" * 60)

    from multic.segmentationschool.Codes.IterativePredict_notebook import predict
    predict(args)

    # Report what was written
    saved = sorted(f for f in os.listdir(args.output_dir) if f.endswith(".json"))
    print("\n" + "=" * 60)
    print(f"Done. {len(saved)} annotation file(s) written to: {args.output_dir}")
    for fname in saved:
        path = os.path.join(args.output_dir, fname)
        with open(path) as fh:
            data = json.load(fh)
        n_elements = len(data.get("elements", []))
        print(f"  {fname:45s}  ({n_elements} regions)")
    print("=" * 60)


if __name__ == "__main__":
    main()
