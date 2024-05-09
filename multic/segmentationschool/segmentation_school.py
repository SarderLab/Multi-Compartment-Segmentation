import os
import argparse
import sys
import time

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

# def get_girder_client(args):
#     gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
#     gc.setToken(args.girderToken)
    
#     return gc

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

    from segmentationschool.Codes.InitializeFolderStructure import initFolder, purge_training_set, prune_training_set
    from segmentationschool.Codes.extract_reference_features import getKidneyReferenceFeatures
    # from TransformXMLs import splice_cortex_XMLs,register_aperio_scn_xmls
    # from randomCropGenerator import randomCropGenerator
    if args.one_network == True:
        from segmentationschool.Codes.IterativeTraining_1X import IterateTraining
        from segmentationschool.Codes.IterativePredict_1X import predict
    else:
        from segmentationschool.Codes.evolve_predictions import evolve
        from segmentationschool.Codes.IterativeTraining import IterateTraining
        from segmentationschool.Codes.IterativePredict import predict

    # for teaching young segmentations networks
    starttime = time.time()
    # if args.project == ' ':
    #     print('Please specify the project name: \n\t--project [folder]')

    if args.option in ['new', 'New']:
        initFolder(args=args)
        savetime(args=args, starttime=starttime)
    elif args.option in ['train', 'Train']:
        IterateTraining(args=args)
        savetime(args=args, starttime=starttime)
    elif args.option in ['predict', 'Predict']:
        predict(args=args)
        savetime(args=args, starttime=starttime)

    elif args.option in ['evolve', 'Evolve']:
        evolve(args=args)
    elif args.option in ['purge', 'Purge']:
        purge_training_set(args=args)
    elif args.option in ['prune', 'Prune']:
        prune_training_set(args=args)
    elif args.option in ['get_features', 'Get_features']:
        getKidneyReferenceFeatures(args=args)
    elif args.option in ['summarize_features', 'Summarize_features']:
        summarizeKidneyReferenceFeatures(args=args)
    elif args.option in ['splice_cortex', 'Splice_cortex']:
        splice_cortex_XMLs(args=args)
    elif args.option in ['register_aperio_scn_xmls', 'Register_aperio_scn_xmls']:
        register_aperio_scn_xmls(args=args)
    elif args.option in ['get_thumbnails', 'Get_thumbnails']:
        from wsi_loader_utils import get_image_thumbnails
        get_image_thumbnails(args)
    elif args.option in ['random_patch_crop', 'random_patch_crop']:
        randomCropGenerator(args=args)

    else:
        print('please specify an option in: \n\t--option [new, train, predict, validate, evolve, purge, prune, get_features, splice_cortex, register_aperio_scn_xmls]')


def savetime(args, starttime):
    if args.option in ['new', 'New']:
        print('new')
        # with open(args.runtime_file, 'w') as timefile:
        #     timefile.write('option' +'\t'+ 'time' +'\t'+ 'epochs_LR' +'\t'+ 'epochs_HR' +'\t'+ 'aug_LR' +'\t'+ 'aug_HR' +'\t'+ 'overlap_percentLR' +'\t'+ 'overlap_percentHR')
    if args.option in ['train', 'Train']:
        print('not much')
        # with open(args.runtime_file, 'a') as timefile:
        #     timefile.write('\n' + args.option +'\t'+ str(time.time()-starttime) +'\t'+ str(args.epoch_LR) +'\t'+ str(args.epoch_HR) +'\t'+ str(args.aug_LR) +'\t'+ str(args.aug_HR) +'\t'+ str(args.overlap_percentLR) +'\t'+ str(args.overlap_percentHR))
    if args.option in ['predict', 'Predict']:
        print('predict')
        # with open(args.runtime_file, 'a') as timefile:
        #     timefile.write('\n' + args.option +'\t'+ str(time.time()-starttime))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ##### Main params (MANDITORY) ##############################################
    # School subject
    parser.add_argument('--girderApiUrl', dest='girderApiUrl', default=' ' ,type=str,
        help='girderApiUrl')
    parser.add_argument('--girderToken', dest='girderToken', default=' ' ,type=str,
        help='girderToken')
    parser.add_argument('--file', dest='file', default=' ' ,type=str,
        help='input WSI file name')
    # option
    parser.add_argument('--option', dest='option', default=' ' ,type=str,
        help='option for [new, train, predict, validate]')
    parser.add_argument('--transfer', dest='transfer', default=' ' ,type=str,
        help='name of project for transfer learning [pulls the newest model]')
    parser.add_argument('--one_network', dest='one_network', default=True ,type=bool,
        help='use only high resolution network for training/prediction/validation')
    parser.add_argument('--target', dest='target', default=None,type=str,
        help='directory with xml transformation targets')
    parser.add_argument('--cortextarget', dest='cortextarget', default=None,type=str,
        help='directory with cortex annotations for splicing')
    parser.add_argument('--output_dir', dest='output_dir', default=None,type=str,
        help='directory to save output excel file')
    parser.add_argument('--wsis', dest='wsis', default=None,type=str,
        help='directory of WSIs for reference feature extraction')
    parser.add_argument('--groupBy', dest='groupBy', default=None,type=str,
        help='Name for histomicsUI converted annotation group')
    parser.add_argument('--patientData', dest='patientData', default=None,type=str,
        help='Location of excel file containing clinical data on patients')
    parser.add_argument('--labelColumns', dest='labelColumns', default=None,type=str,
        help='Column in excel file to use as label')
    parser.add_argument('--labelModality', dest='labelModality', default=None,type=str,
        help='Column in excel file to use as label')
    parser.add_argument('--IDColumn', dest='IDColumn', default='Label_slides',type=str,
        help='Excel column with file name links')
    parser.add_argument('--plotFill', dest='plotFill', default=True,type=str2bool,
        help='Excel column with file name links')
    parser.add_argument('--scatterFeatures', dest='scatterFeatures', default='5,6',type=str,
        help='Excel column with file name links')
    parser.add_argument('--anchor', dest='anchor', default='Age',type=str,
        help='Biometric link data for scatterplot')
    parser.add_argument('--exceloutfile', dest='exceloutfile', default=None,type=str,
        help='Name of output excel file for feature aggregation')


# args.huelabel,args.rowlabel,args.binRows
    parser.add_argument('--SummaryOption', dest='SummaryOption', default=None,type=str,
        help='What type of feature summary to generate, options:\n'+
            'BLDensity,ULDensity,UDensity,BDensity,standardScatter,anchorScatter')

    # automatically generated
    parser.add_argument('--base_dir', dest='base_dir', default=os.getcwd(),type=str,
        help='base directory of Data folder')

    parser.add_argument('--code_dir', dest='code_dir', default=os.getcwd(),type=str,
        help='base directory of code folder')


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
        help='downsample mask to find usable regions')
    #Low resolution parameters
    parser.add_argument('--overlap_percentLR', dest='overlap_percentLR', default=0.5 ,type=float,
        help='overlap percentage of low resolution blocks [0-1]')
    parser.add_argument('--boxSizeLR', dest='boxSizeLR', default=450 ,type=int,
        help='size of low resolution blocks')
    parser.add_argument('--downsampleRateLR', dest='downsampleRateLR', default=16 ,type=int,
        help='reduce image resolution to 1/downsample rate')
    #High resolution parameters
    parser.add_argument('--overlap_percentHR', dest='overlap_percentHR', default=0 ,type=float,
        help='overlap percentage of high resolution blocks [0-1]')
    parser.add_argument('--boxSize', dest='boxSize', default=2048 ,type=int,
        help='size of high resolution blocks')
    parser.add_argument('--downsampleRateHR', dest='downsampleRateHR', default=1 ,type=int,
        help='reduce image resolution to 1/downsample rate')
    parser.add_argument('--training_max_size', dest='training_max_size', default=512 ,type=int,
        help='padded region for low resolution region extraction')
    parser.add_argument('--Mag20X', dest='Mag20X', default=False,type=str2bool,
        help='Perform prediction for 20X (true) slides rather than 40X (false)')

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
    parser.add_argument('--CNNbatch_sizeHR', dest='CNNbatch_sizeHR', default=2 ,type=int,
        help='Size of batches for training high resolution CNN')
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
    parser.add_argument('--chop_data', dest='chop_data', default='false',
        type=str, help='chop and augment new data before training')
    parser.add_argument('--crop_detectron_trainset', dest='crop_detectron_trainset', default=False,type=str2bool,
        help='chop dot based images to this max size')
    parser.add_argument('--predict_data', dest='predict_data', default=True,type=str2bool,
        help='chop dot based images to this max size')
    parser.add_argument('--roi_thresh', dest='roi_thresh', default=0.01,type=float,
        help='chop dot based images to this max size')

    ### Params for saving results ###
    parser.add_argument('--outDir', dest='outDir', default='Predictions' ,type=str,
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
    parser.add_argument('--min_size', dest='min_size', default=[30,30,30,30,30,30] ,type=int,
        help='min size region to be considered after prepass [in pixels]')
    parser.add_argument('--bordercrop', dest='bordercrop', default=300 ,type=int,
        help='min size region to be considered after prepass [in pixels]')
    parser.add_argument('--LR_region_pad', dest='LR_region_pad', default=50 ,type=int,
        help='padded region for low resolution region extraction')
    parser.add_argument('--show_interstitium', dest='show_interstitium', default=True ,type=str2bool,
        help='padded region for low resolution region extraction')
    
    parser.add_argument('--xml_path', dest='xml_path', default=' ' ,type=str,
        help='path to xml file')

    parser.add_argument('--ext', dest='ext', default='.svs' ,type=str,
        help='file extention')

    parser.add_argument('--platform', dest='platform', default='DSA' ,type=str,
        help='Run Platform, HPG or DSA')

    parser.add_argument('--item_id', dest='item_id', default=' ' , type=str,
        help='item id of the WSI in DSA')




    args = parser.parse_args()
    main(args=args)
