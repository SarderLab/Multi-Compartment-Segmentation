import sys
import os, girder_client
import numpy as np
import pandas as pd
from tiffslide import TiffSlide
from ctk_cli import CLIArgumentParser

sys.path.append("..")

from segmentationschool.extraction_utils.extract_ffpe_features import xml_to_mask
from segmentationschool.extraction_utils.layer_dict import NAMES_DICT
from segmentationschool.extraction_utils.process_mc_features import process_glom_features, process_tubules_features, process_arteriol_features


MODx=np.zeros((3,))
MODy=np.zeros((3,))
MODz=np.zeros((3,))
MODx[0]= 0.644211
MODy[0]= 0.716556
MODz[0]= 0.266844

MODx[1]= 0.175411
MODy[1]= 0.972178
MODz[1]= 0.154589

MODx[2]= 0.0
MODy[2]= 0.0
MODz[2]= 0.0
MOD=[MODx,MODy,MODz]
NAMES = ['non_globally_sclerotic_glomeruli','globally_sclerotic_glomeruli','tubules','arteries/arterioles']


def main(args):

    file = args.input_file
    _ = os.system("printf 'Using data from girder_client file: {}\n'".format(file))
    file_name = file.split('/')[-1]
    plain_name = file_name.split('.')[0]
    folder = args.base_dir
    base_dir_id = folder.split('/')[-2]
    _ = os.system("printf '\nUsing data from girder_client Folder: {}\n'".format(folder))

    
    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)

    files = list(gc.listItem(base_dir_id))
    # dict to link filename to gc id
    item_dict = dict()
    for file in files:
        d = {file['name']:file['_id']}
        item_dict.update(d)
    
    file_id = item_dict[file_name]

    annotations = gc.get('/annotation/item/{}'.format(file_id))

    annotations_filtered = [annot for annot in annotations if annot['annotation']['name'].strip() in NAMES]

    del annotations

    cwd = os.getcwd()
    print(cwd)

    slide = TiffSlide(args.input_file)
    x,y = slide.dimensions

    mpp = slide.properties['tiffslide.mpp-x']
    mask_xml = xml_to_mask(annotations_filtered,(0,0),(x,y),downsample_factor=args.downsample_factor)

    gloms = process_glom_features(mask_xml, NAMES_DICT['non_globally_sclerotic_glomeruli'], MOD, slide,mpp, h_threshold=args.h_threshold, saturation_threshold=args.saturation_threshold)
    s_gloms = process_glom_features(mask_xml, NAMES_DICT['globally_sclerotic_glomeruli'], MOD, slide,mpp, h_threshold=args.h_threshold, saturation_threshold=args.saturation_threshold)
    tubs = process_tubules_features(mask_xml, NAMES_DICT['tubules'], MOD, slide,mpp,whitespace_threshold=args.whitespace_threshold)
    arts = process_arteriol_features(mask_xml, NAMES_DICT['arteries/arterioles'], mpp)


    all_comparts = [gloms,s_gloms,tubs, arts]
    all_columns = [['x1','x2','y1','y2','Area','Mesangial Area','Mesangial Fraction'],
                   ['x1','x2','y1','y2','Area','Mesangial Area','Mesangial Fraction'],
                   ['x1','x2','y1','y2','Average TBM Thickness','Average Cell Thickness','Luminal Fraction'],
                   ['x1','x2','y1','y2','Arterial Area']]
    compart_names = ['gloms','s_gloms','tubs','arts']
    
    _ = os.system("printf '\tWriting Excel file: [{}]\n'".format(args.output_filename))
    with pd.ExcelWriter(args.output_filename) as writer:
        for idx,compart in enumerate(all_comparts):
            df = pd.DataFrame(compart,columns=all_columns[idx])
            df.to_excel(writer, index=False, sheet_name=compart_names[idx])

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
