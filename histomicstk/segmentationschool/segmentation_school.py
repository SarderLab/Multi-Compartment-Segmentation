import os
import argparse
import sys
import numpy as np
import time

sys.path.append(os.getcwd()+'/Codes')


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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):

    from InitializeFolderStructure import initFolder, purge_training_set, prune_training_set
    from TransformXMLs import transform_XMLs
    # from extract_reference_features import extractKidneyReferenceFeatures
    if args.one_network == True:
        from IterativeTraining_1X_chopless_test import IterateTraining
        from IterativePredict_1X import predict, validate
    else:
        from evolve_predictions import evolve
        from IterativeTraining import IterateTraining
        from IterativePredict import predict, validate

    # for teaching young segmentations networks
    starttime = time.time()
    if args.project == ' ':
        print('Please specify the project name: \n\t--project [folder]')

    elif args.option in ['new', 'New']:
        initFolder(args=args)
        savetime(args=args, starttime=starttime)
    elif args.option in ['train', 'Train']:
        IterateTraining(args=args)
        savetime(args=args, starttime=starttime)
    elif args.option in ['predict', 'Predict']:
        predict(args=args)
        savetime(args=args, starttime=starttime)
    elif args.option in ['validate', 'Validate']:
        validate(args=args)
    elif args.option in ['evolve', 'Evolve']:
        evolve(args=args)
    elif args.option in ['purge', 'Purge']:
        purge_training_set(args=args)
    elif args.option in ['prune', 'Prune']:
        prune_training_set(args=args)
    elif args.option in ['transform_xmls', 'Transform_xmls']:
        transform_XMLs(args=args)
    elif args.option in ['extract_features', 'Extract_features']:
        extractKidneyReferenceFeatures(args=args)

    else:
        print('please specify an option in: \n\t--option [new, train, predict, validate]')


def savetime(args, starttime):
    if args.option in ['new', 'New']:
        with open(args.base_dir + '/' + args.project + '/runtime.txt', 'w') as timefile:
            timefile.write('option' +'\t'+ 'time' +'\t'+ 'epochs_LR' +'\t'+ 'epochs_HR' +'\t'+ 'aug_LR' +'\t'+ 'aug_HR' +'\t'+ 'overlap_percentLR' +'\t'+ 'overlap_percentHR')
    if args.option in ['train', 'Train']:
        with open(args.base_dir + '/' + args.project + '/runtime.txt', 'a') as timefile:
            timefile.write('\n' + args.option +'\t'+ str(time.time()-starttime) +'\t'+ str(args.epoch_LR) +'\t'+ str(args.epoch_HR) +'\t'+ str(args.aug_LR) +'\t'+ str(args.aug_HR) +'\t'+ str(args.overlap_percentLR) +'\t'+ str(args.overlap_percentHR))
    if args.option in ['predict', 'Predict']:
        with open(args.base_dir + '/' + args.project + '/runtime.txt', 'a') as timefile:
            timefile.write('\n' + args.option +'\t'+ str(time.time()-starttime))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ##### Main params (MANDITORY) ##############################################
    # School subject
    parser.add_argument('--project', dest='project', default=' ' ,type=str,
        help='Starting directory to contain training project')
    # option
    parser.add_argument('--option', dest='option', default=' ' ,type=str,
        help='option for [new, train, predict, validate]')
    parser.add_argument('--transfer', dest='transfer', default=' ' ,type=str,
        help='name of project for transfer learning [pulls the newest model]')
    parser.add_argument('--one_network', dest='one_network', default=True ,type=bool,
        help='use only high resolution network for training/prediction/validation')

    # automatically generated
    parser.add_argument('--base_dir', dest='base_dir', default=os.getcwd(),type=str,
        help='base directory of code folder')
    parser.add_argument('--target', dest='target', default=None,type=str,
        help='directory to transform xmls')

    ##### Args for training / prediction ####################################################
    parser.add_argument('--gpu_num', dest='gpu_num', default=2 ,type=int,
        help='number of GPUs avalable')
    parser.add_argument('--gpu', dest='gpu', default=0 ,type=int,
        help='GPU to use for prediction')
    parser.add_argument('--iteration', dest='iteration', default='none' ,type=str,
        help='Which iteration to use for prediction')
    parser.add_argument('--prune_HR', dest='prune_HR', default=0.0 ,type=float,
        help='percent of high rez data to be randomly removed [0-1]-->[none-all]')
    parser.add_argument('--prune_LR', dest='prune_LR', default=0.0 ,type=float,
        help='percent of low rez data to be randomly removed [0-1]-->[none-all]')
    parser.add_argument('--classNum', dest='classNum', default=0 ,type=int,
        help='number of classes present in the training data plus one (one class is specified for background)')
    parser.add_argument('--classNum_HR', dest='classNum_HR', default=0 ,type=int,
        help='number of classes present in the High res training data [USE ONLY IF DIFFERENT FROM LOW RES]')
    parser.add_argument('--modelfile', dest='modelfile', default=None ,type=str,
        help='the desired model file to use for training or prediction')

    ### Params for cutting wsi ###
    #White level cutoff
    parser.add_argument('--white_percent', dest='white_percent', default=0.01 ,type=float,
        help='white level checkpoint for chopping')
    parser.add_argument('--chop_thumbnail_resolution', dest='chop_thumbnail_resolution', default=16,type=int,
        help='Amount of downsampling in each dimension to determine usable tissue regions')
    #Low resolution parameters
    parser.add_argument('--overlap_percentLR', dest='overlap_percentLR', default=0.5 ,type=float,
        help='overlap percentage of low resolution blocks [0-1]')
    parser.add_argument('--boxSizeLR', dest='boxSizeLR', default=450 ,type=int,
        help='size of low resolution blocks')
    parser.add_argument('--downsampleRateLR', dest='downsampleRateLR', default=16 ,type=int,
        help='reduce image resolution to 1/downsample rate')
    #High resolution parameters
    parser.add_argument('--overlap_rate', dest='overlap_rate', default=0.5 ,type=float,
        help='overlap percentage of high resolution blocks [0-1]')
    parser.add_argument('--boxSize', dest='boxSize', default=1200 ,type=int,
        help='size of high resolution blocks')
    parser.add_argument('--downsampleRate', dest='downsampleRate', default=1 ,type=int,
        help='reduce image resolution to 1/downsample rate')
    parser.add_argument('--training_max_size', dest='training_max_size', default=512 ,type=int,
        help='padded region for low resolution region extraction')
    parser.add_argument('--box_supervision', dest='box_supervision', default=True,type=str2bool,
        help='Use rectangle annotations to confine chopping')
    parser.add_argument('--chop_with_replacement', dest='chop_with_replacement', default=False,type=str2bool,
        help='make ultimate contour class ID equal to dot-based ID')
    parser.add_argument('--standard_chop', dest='standard_chop', default=True,type=str2bool,
        help='use contour class ID as defined by the region itself')
    parser.add_argument('--get_new_tissue_masks', dest='get_new_tissue_masks', default=False,type=str2bool,
        help="Don't load usable tisse regions from disk, create new ones")
    parser.add_argument('--balanceClasses', dest='balanceClasses', default='3,4,6',type=str,
        help="which classes to balance during training")

    ### Params for augmenting data ###
    #High resolution
    parser.add_argument('--aug_HR', dest='aug_HR', default=3 ,type=int,
        help='augment high resolution set this many magnitudes')
    #Low resolution
    parser.add_argument('--aug_LR', dest='aug_LR', default=15 ,type=int,
        help='augment low resolution set this many magnitudes')
    #Color space transforms
    parser.add_argument('--hbound', dest='hbound', default=0.01 ,type=float,
        help='Gaussian variance defining bounds on Hue shift for HSV color augmentation')
    parser.add_argument('--lbound', dest='lbound', default=0.025 ,type=float,
        help='Gaussian variance defining bounds on L* gamma shift for color augmentation [alters brightness/darkness of image]')

    ### Params for training networks ###
    #Low resolution hyperparameters
    parser.add_argument('--CNNbatch_sizeLR', dest='CNNbatch_sizeLR', default=2 ,type=int,
        help='Size of batches for training low resolution CNN')
    #High resolution hyperparameters
    parser.add_argument('--batch_size', dest='batch_size', default=3 ,type=int,
        help='Size of batches for training high resolution CNN')
    parser.add_argument('--train_steps', dest='train_steps', default=300000 ,type=int,
        help='Size of batches for training high resolution CNN')
    #Hyperparameters
    #Hyperparameters
    parser.add_argument('--epoch_LR', dest='epoch_LR', default=1 ,type=int,
        help='training epochs for low resolution network')
    parser.add_argument('--epoch_HR', dest='epoch_HR', default=1 ,type=int,
        help='training epochs for high resolution network')
    parser.add_argument('--saveIntervals', dest='saveIntervals', default=10 ,type=int,
        help='how many checkpoints get saved durring training')
    parser.add_argument('--learning_rate_HR', dest='learning_rate_HR', default=2.5e-4,
        type=float, help='High rez learning rate')
    parser.add_argument('--learning_rate_LR', dest='learning_rate_LR', default=2.5e-4,
        type=float, help='Low rez learning rate')

    parser.add_argument('--crop_detectron_trainset', dest='crop_detectron_trainset', default=False,type=str2bool,
        help='chop dot based images to this max size')
    parser.add_argument('--predict_data', dest='predict_data', default=True,type=str2bool,
        help='chop dot based images to this max size')
    parser.add_argument('--roi_thresh', dest='roi_thresh', default=0.7,type=float,
        help='chop dot based images to this max size')
    parser.add_argument('--prepare_detectron_json', dest='prepare_detectron_json', default=True,type=str2bool,
        help='chop dot based images to this max size')
    parser.add_argument('--custom_image_means', dest='custom_image_means', default=False,type=str2bool,
        help='measure image mean for network training')
    parser.add_argument('--check_training_data', dest='check_training_data', default=False,type=str2bool,
        help='check images visually before training')
    parser.add_argument('--hsv_aug_prob', dest='hsv_aug_prob', default=0.1,type=float,
        help='if rand (0,1) > hsv_aug_prob, apply aug to this image')

    ### Params for saving results ###
    parser.add_argument('--outDir', dest='outDir', default='/Predictions/' ,type=str,
        help='output directory')
    parser.add_argument('--save_outputs', dest='save_outputs', default=False ,type=bool,
        help='save outputs from chopping etc. [final image masks]')
    parser.add_argument('--imBoxExt', dest='imBoxExt', default='.jpeg' ,type=str,
        help='ext of saved image blocks')
    parser.add_argument('--finalImgExt', dest='finalImgExt', default='.jpeg' ,type=str,
        help='ext of final saved images')
    parser.add_argument('--wsi_ext', dest='wsi_ext', default='.svs,.scn,.ndpi' ,type=str,
        help='file ext of wsi images')
    parser.add_argument('--bg_intensity', dest='bg_intensity', default=.5 ,type=float,
        help='if displaying output classifications [save_outputs = True] background color [0-1]')
    parser.add_argument('--approximation_downsample', dest='approx_downsample', default=1 ,type=float,
        help='Amount to downsample high resolution prediction boundaries for smoothing')


    ### Params for optimizing wsi mask cleanup ###
    parser.add_argument('--min_size', dest='min_size', default=[0,30,30,30,30] ,type=int,
        help='min size region to be considered after prepass [in pixels]')
    parser.add_argument('--bordercrop', dest='bordercrop', default=200 ,type=int,
        help='min size region to be considered after prepass [in pixels]')
    parser.add_argument('--LR_region_pad', dest='LR_region_pad', default=50 ,type=int,
        help='padded region for low resolution region extraction')



    args = parser.parse_args()
    main(args=args)
