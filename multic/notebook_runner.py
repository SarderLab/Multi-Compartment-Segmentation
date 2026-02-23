import argparse
import os
import sys

DEFAULT_VALS = {
    'option': 'predict',
    'white_percent': 0.01,
    'chop_thumbnail_resolution': 16,
    'overlap_percentHR': 0,
    'boxSize': 2048,
    'downsampleRateHR': 1,
    'Mag20X': False,
    'roi_thresh': 0.01,
    'min_size': [30, 30, 30, 30, 30, 30],
    'bordercrop': 300,
    'show_interstitium': True
}


def run_notebook(input_file: str, modelfile: str, output_dir: str, **kwargs):
    """
    Run Multi-Compartment Segmentation without Girder.

    Segments 6 Functional Tissue Units from a whole slide image and saves
    per-class JSON annotation files into output_dir.

    Parameters
    ----------
    input_file : str
        Path to the whole slide image (WSI) file (.svs, .tif, .scn, etc.)
    modelfile : str
        Path to the Detectron2 model weights (.pth)
    output_dir : str
        Directory where JSON annotation files will be saved.
        One file per compartment class, e.g. cortical_interstitium.json
    **kwargs
        Optional overrides for any segmentation parameter (see DEFAULT_VALS).
        Examples:
            boxSize=1024, roi_thresh=0.05, show_interstitium=False

    Example
    -------
    from multic.notebook_runner import run_notebook

    run_notebook(
        input_file='/data/slides/sample.svs',
        modelfile='/data/models/FUSION_MCS_FFPE_v1.pth',
        output_dir='/data/outputs/sample/'
    )
    """
    # Ensure segmentationschool package is importable
    _pkg_dir = os.path.dirname(__file__)
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)

    from multic.segmentationschool.Codes.IterativePredict_notebook import predict

    args = argparse.Namespace()
    args.input_file = input_file
    args.modelfile = modelfile
    args.output_dir = output_dir
    args.file = input_file  # predict() reads args.file for the slide path

    for key, val in DEFAULT_VALS.items():
        setattr(args, key, val)
    for key, val in kwargs.items():
        setattr(args, key, val)

    predict(args)
