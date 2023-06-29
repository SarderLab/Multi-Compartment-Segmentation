import os, girder_client
import numpy as np

from scipy.ndimage.morphology import distance_transform_edt
from tqdm import tqdm
import tiffslide as openslide
from skimage.color import rgb2lab
from skimage import exposure
from skimage.morphology import remove_small_objects,binary_erosion,binary_dilation,disk,binary_opening,binary_closing
from skimage.filters import *
import pandas as pd
from skimage import measure
from scipy.ndimage import label
from histomicstk.cli.utils import CLIArgumentParser

import sys
sys.path.append("..")

from segmentationschool.extraction_utils.extract_ffpe_features import imreconstruct, xml_to_mask
from segmentationschool.extraction_utils.PAS_deconvolution import deconvolution,process_glom_features

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

    annotations_1 = [annot for annot in annotations if annot['annotation']['name'].strip() in NAMES]

    del annotations

    cwd = os.getcwd()
    print(cwd)

    slide = openslide.OpenSlide(args.input_file)
    x,y = slide.dimensions

    mpp = slide.properties['tiffslide.mpp-x']
    mask_xml = xml_to_mask(annotations_1,(0,0),(x,y),downsample_factor=args.downsample_factor)

    gloms = process_glom_features(mask_xml,3,MOD,slide,mpp)
    s_gloms = process_glom_features(mask_xml,4,MOD,slide,mpp)

    tubules = mask_xml == 5
    tubules = tubules.astype(np.uint8)
    tubules = measure.label(tubules)
    tubules_unique_max = np.max(tubules)
    tubs = np.zeros((tubules_unique_max,7))
    props = measure.regionprops(tubules)

    for i in tqdm(range(tubules_unique_max),desc='Tubules'):
        #Area, Cellularity, Mesangial Area to Cellularity Ratio
        # i=2615
        area = (props[i].area)*(mpp**2)
        x1,y1,x2,y2 = props[i].bbox


        crop = slide.read_region((y1,x1),0,(y2-y1,x2-x1))
        crop = np.array(crop)
        crop = crop[:,:,:3]

        mask = (tubules[x1:x2,y1:y2] == i+1).astype(np.uint8)

        lab = rgb2lab(crop)
        lab = lab[:,:,0]
        lab = lab/100
        WS = lab > 0.88
        WS = remove_small_objects(WS,100)
        WS = binary_opening(WS,disk(1))
        WS[mask==0] = 0

        _,pas,_ = deconvolution(crop,MOD)
        pas = 255-pas
        pas = exposure.adjust_gamma(pas, 3)
        pas = (pas > threshold_otsu(pas))
        pas = binary_closing(pas,disk(1))
        pas = pas.astype(np.uint8)

        boundary_w_mem = binary_dilation(mask,disk(10))
        blim = boundary_w_mem
        indel = binary_erosion(boundary_w_mem,disk(10))
        blim[indel >0] = 0
        boundary_w_mem = boundary_w_mem.astype(np.uint8)
        indel = indel.astype(np.uint8)
        blim = blim.astype(np.uint8)

        cons_tbm = ((blim + pas) > 0).astype(np.uint8)

        # boundary_w_mem[indel>0] = 0

        tbm = imreconstruct(boundary_w_mem,pas)
        tbm[boundary_w_mem==0] = 0
        tbm = tbm >0
        tbm = remove_small_objects(tbm,50)
        tbm = binary_closing(tbm,disk(1))



        tbmdist = distance_transform_edt(tbm)
        tbm = tbm.astype(np.uint8)
        tbm_l = measure.label(tbm)
        tbm_props = measure.regionprops(tbm_l)

        tbm_areas = []
        tbm_thicknesses = []
        for j in range(len(tbm_props)):
            tbm_areas.append(tbm_props[j].area)
            tbm_thicknesses.append(tbm_props[j].area/((tbm_props[j].perimeter)/2))

        area_tot = np.sum(np.array(tbm_areas))
        tbm_avg = np.sum(np.multiply(np.array(tbm_thicknesses),(np.array(tbm_areas)/area_tot)))


        WS = WS.astype(np.uint8)

        cyto = mask - WS - tbm
        cyto = cyto > 0
        cyto = remove_small_objects(cyto,50)
        cyto = cyto.astype(np.uint8)

        cyto_l = measure.label(cyto)
        cyto_props = measure.regionprops(cyto_l)

        cyto_areas = []
        cyto_thicknesses = []
        for j in range(len(cyto_props)):
            cyto_areas.append(cyto_props[j].area)
            cyto_thicknesses.append(cyto_props[j].area/((cyto_props[j].perimeter)/2))

        area_tot = np.sum(np.array(cyto_areas))
        cyto_avg = np.sum(np.multiply(np.array(cyto_thicknesses),(np.array(cyto_areas)/area_tot)))
        
        tubs[i,0] = x1
        tubs[i,1] = x2
        tubs[i,2] = y1
        tubs[i,3] = y2
        tubs[i,4] = tbm_avg*(mpp**2)
        tubs[i,5] = cyto_avg*(mpp**2)
        tubs[i,6] = np.sum(WS) / np.sum(mask)#

    del tubules

    arteries = mask_xml == 6
    arteries = arteries.astype(np.uint8)
    arteries = measure.label(arteries)
    arts_unique_max = np.max(arteries)
    arts = np.zeros((arts_unique_max,5))
    props = measure.regionprops(arteries)

    for i in tqdm(range(arts_unique_max),desc='Arteries'):
        area = (props[i].area)*(mpp**2)
        x1,y1,x2,y2 = props[i].bbox

        arts[i,0] = x1
        arts[i,1] = x2
        arts[i,2] = y1
        arts[i,3] = y2
        arts[i,4] = area

    del arteries

    all_comparts = [gloms,s_gloms,tubs, arts]
    all_columns = [['x1','x2','y1','y2','Area','Mesangial Area','Mesangial Fraction'],
                   ['x1','x2','y1','y2','Area','Mesangial Area','Mesangial Fraction'],
                   ['x1','x2','y1','y2','Average TBM Thickness','Average Cell Thickness','Luminal Fraction'],
                   ['x1','x2','y1','y2','Arterial Area']]
    compart_names = ['gloms','s_gloms','tubs','arts']
    

    # gloms.to_csv(plain_name+'_gloms.csv')
    # s_gloms.to_csv(plain_name+'_s_gloms.csv')
    # tubs.to_csv(plain_name+'_tubs.csv')
    # arts.to_csv(plain_name+'_arteries.csv')

    _ = os.system("printf '\tWriting Excel file: [{}]\n'".format(args.output_filename))
    with pd.ExcelWriter(args.output_filename) as writer:
        for idx,compart in enumerate(all_comparts):
            df = pd.DataFrame(compart,columns=all_columns[idx])
            df.to_excel(writer, index=False, sheet_name=compart_names[idx])

if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
