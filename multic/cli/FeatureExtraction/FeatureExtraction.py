import sys
import os
import numpy as np
import pandas as pd
from tiffslide import TiffSlide
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
# from ctk_cli import CLIArgumentParser

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
# NAMES = ['non_globally_sclerotic_glomeruli','globally_sclerotic_glomeruli','tubules','arteries/arterioles']
NAMES = [3, 4, 5, 6]

def main():
    
    parser = ArgumentParser(description="Input file")
    
    parser.add_argument("--input_dir", type=str, help="base dir")
    parser.add_argument("--output_dir", type=str, help="output dir")
    parser.add_argument("--downsample_factor", type=float, default=1.0, help="downsample factor")    
    parser.add_argument("--h_threshold", type=float, default=160, help="h threshold")
    parser.add_argument("--whitespace_threshold", type=float, default=0.88, help="whitespace threshold")
    parser.add_argument("--saturation_threshold", type=float, default=0.3, help="saturation threshold")
    
    args = parser.parse_args()

    image_files = os.listdir(args.input_dir)
    
    for filename in image_files:
        if filename.endswith('.svs') == False:
            continue        
        image_file = os.path.join(args.input_dir, filename.split('.')[0] + '.svs')        
        xml_file   = os.path.join(args.input_dir, filename.split('.')[0] + '.xml')
        
        # print(image_file)
        # print(xml_file)
        
        xmlRoot = ET.parse(xml_file).getroot()
        
        annotations = xmlRoot.findall('.//Annotation')                

        annotations_filtered = [annot for annot in annotations if annot.get('Id') in map(str, NAMES)]

        del annotations

        cwd = os.getcwd()
        print(cwd)

        slide = TiffSlide(image_file)
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
        
        output_filename = os.path.join(args.output_dir, image_file.split('.')[0] + '.xlsx')
        print("printf '\tWriting Excel file: [{}]\n'".format(output_filename))
        with pd.ExcelWriter(args.output_filename) as writer:
            for idx,compart in enumerate(all_comparts):
                df = pd.DataFrame(compart,columns=all_columns[idx])
                df.to_excel(writer, index=False, sheet_name=compart_names[idx])

if __name__ == "__main__":
    main()