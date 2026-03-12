import os
import sys
import argparse
sys.path.append('..')
from segmentationschool.Codes.IterativePredict_notebook import predict


DEFAULT_VALS = {
    'option': 'predict_notebook',
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


def main(args):
    for d in DEFAULT_VALS:
        if d not in vars(args):
            setattr(args, d, DEFAULT_VALS[d])

    setattr(args, 'file', args.input_file)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Running on: {args.input_file}')
    print(f'Model: {args.modelfile}')
    print(f'Output dir: {args.output_dir}')

    for d in vars(args):
        print(f'argument: {d}, value: {getattr(args, d)}')

    predict(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi Compartment Segmentation (Girder-free, for Jupyter notebooks)'
    )
    parser.add_argument('--input_file', required=True,
                        help='Path to the whole slide image (WSI) file')
    parser.add_argument('--modelfile', required=True,
                        help='Path to the Detectron2 model weights (.pth)')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where per-class JSON annotation files will be saved')
    args = parser.parse_args()
    main(args)
