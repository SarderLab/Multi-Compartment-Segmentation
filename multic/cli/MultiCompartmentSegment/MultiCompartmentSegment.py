import os
import sys
from ctk_cli import CLIArgumentParser
sys.path.append('..')
from segmentationschool.segmentation_school import run_it


DEFAULT_VALS = {
        'girderApiUrl':' ',
        'girderToken':' ',
        'files':' ',
        'option':'predict',
        'transfer':' ',
        'one_network':True,
        'target':None,
        'cortextarget':None,
        'output':None,
        'wsis':None,
        'groupBy':None,
        'patientData':None,
        'labelColumns':None,
        'labelModality':None,
        'IDColumn':'Label_slides',
        'plotFill':True,
        'scatterFreatures':'5,6',
        'anchor':'Age',
        'exceloutfile':None,
        'SummaryOption':None,
        'base_dir': os.getcwd(),
        'code_dir':os.getcwd(),
        'gpu_num':2,
        'gpu':0,
        'iteration':'none',
        'prune_HR':0.0,
        'prune_LR':0.0,
        'classNum':0,
        'classNum_HR':0,
        'modelfile':None,
        'white_percent':0.01,
        'chop_thumbnail_resolution':16,
        'overlap_percentLR':0.5,
        'boxSizeLR':450,
        'downsampleRateLR':16,
        'overlap_percentHR':0,
        'boxSize':2048,
        'downsampleRateHR':1,
        'training_max_size':512,
        'Mag20X':False,
        'aug_HR':3,
        'aug_LR':15,
        'hbound':0.01,
        'lbound':0.025,
        'CNNbatch_sizeLR':2,
        'CNNbatch_sizeHR':2,
        'epoch_LR':1,
        'epoch_HR':1,
        'saveIntervals':10,
        'learning_rate_HR':2.5e-4,
        'learning_rate_LR':2.53-4,
        'chop_data':'false',
        'crop_detectron_trainset':False,
        'predict_data':True,
        'roi_thresh':0.01,
        'outDir':'Predictions',
        'save_outputs':False,
        'imBoxExt':'.jpeg',
        'finalImgExt':'.jpeg',
        'wsi_ext':'.svs,.scn,.ndpi',
        'bg_intensity':0.5,
        'approx_downsample':1,
        'min_size':[30,30,30,30,30,30],
        'bordercrop':300,
        'LR_region_pad':50,
        'show_interstitium':True
    }


def main(args):

    folder = args.base_dir
    base_dir_id = folder.split('/')[-2]
    _ = os.system("printf '\nUsing data from girder_client Folder: {}\n'".format(folder))
    print('new version')
    _ = os.system("printf '\n---\n\nFOUND: [{}]\n'".format(args.files))

    cwd = os.getcwd()
    print(cwd)
    os.chdir(cwd)

    for d in DEFAULT_VALS:
        if d not in list(vars(args).keys()):
            setattr(args,d,DEFAULT_VALS[d])

    print(vars(args))
    for d in vars(args):
        print(f'argument: {d}, value: {getattr(args,d)}')

    run_it(args)

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
