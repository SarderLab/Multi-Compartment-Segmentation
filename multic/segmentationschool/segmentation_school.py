import os
import argparse
import sys

sys.path.append('..')


"""
main code for training semantic segmentation of WSI iteratively

    --option              -   code options
        [new]             -   set up a new project
        [train]           -   begin network training with new data
        [predict]         -   use trained network to annotate new data
        [validate]        -   get the network performance on holdout dataset
        [evolve]          -   visualize the evolving network predictions
        [purge]           -   remove previously chopped/augmented data from project
        [prune]           -   randomly remove saved training images (--prune_HR/LR)

    --project
        [<project name>]  -   specify the project name

    --transfer
        [<project name>]  -   pull newest model from specified project
                            for transfer learning

"""
def main(args):

    from segmentationschool.Codes.IterativeTraining_1X import IterateTraining
    from segmentationschool.Codes.IterativePredict_1X import predict

    if args.option in ['train', 'Train']:
        IterateTraining(args=args)

    elif args.option in ['predict', 'Predict']:
        predict(args=args)

    else:
        print('please specify an option in: \n\t--option [predict or train]')

# importable function
def run_it(args):

    main(args)
