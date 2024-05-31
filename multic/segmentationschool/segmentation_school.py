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
    # from extract_reference_features import getKidneyReferenceFeatures,summarizeKidneyReferenceFeatures
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


# importable function
def run_it(args):

    main(args)
