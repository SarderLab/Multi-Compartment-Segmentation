import numpy as np
from tqdm import tqdm
from skimage.color import rgb2lab
from skimage import exposure
from skimage.morphology import remove_small_objects,binary_erosion,binary_dilation,disk,binary_opening,binary_closing
from skimage.filters import threshold_otsu
from skimage import measure

from .extract_ffpe_features import imreconstruct
from .PAS_deconvolution import deconvolution

GLOM_DICT = {3:'Glomeruli',4:'Sclerotic Glomeruli'}

def process_glom_features(mask_xml, glom_value, MOD, slide, mpp, h_threshold, satruation_threshold):

    glomeruli = mask_xml == glom_value
    glomeruli = glomeruli.astype(np.uint8)
    glomeruli = measure.label(glomeruli)
    glomeruli_unique_max = np.max(glomeruli)
    gloms = np.zeros((glomeruli_unique_max,7))
    props = measure.regionprops(glomeruli)

    for i in tqdm(range(glomeruli_unique_max),desc=GLOM_DICT[glom_value]):
        #Area, Cellularity, Mesangial Area to Cellularity Ratio
        area = (props[i].area)*(mpp**2)
        x1,y1,x2,y2 = props[i].bbox

        crop = slide.read_region((y1,x1),0,(y2-y1,x2-x1))
        crop = np.array(crop)
        crop = crop[:,:,:3]

        mask = (glomeruli[x1:x2,y1:y2] == i+1).astype(np.uint8)

        hsv = rgb2hsv(crop)
        hsv = hsv[:,:,1]
        pas_seg = hsv > satruation_threshold
        pas_seg = pas_seg.astype(np.uint8)
        pas_seg = np.multiply(pas_seg,mask)
        pas_seg = pas_seg.astype(np.uint8)

        h,_,_ = deconvolution(crop,MOD)

        h = 255-h
        # h = (h>threshold_otsu(h))
        h = h>h_threshold
        h = h.astype(np.uint8)


        pas_seg = ((pas_seg - h) >0).astype(np.uint8)

        mask_pixels = np.sum(mask)
        pas_pixels = np.sum(pas_seg)

        mes_fraction = (pas_pixels*(mpp**2)) / area

        mes_fraction = pas_pixels/mask_pixels

        gloms[i,0] = x1
        gloms[i,1] = x2
        gloms[i,2] = y1
        gloms[i,3] = y2
        gloms[i,4] = area*(mpp**2)
        gloms[i,5] = pas_pixels*(mpp**2)
        gloms[i,6] = mes_fraction
    
    del glomeruli

    return gloms

def process_tubules_features(mask_xml, tub_value, MOD, slide, mpp, whitespace_threshold):

    tubules = mask_xml == tub_value
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
        WS = lab > whitespace_threshold
        WS = remove_small_objects(WS,100)
        WS = binary_opening(WS,disk(1))
        WS[mask==0] = 0

        _,pas,_ = deconvolution(crop,MOD)
        pas = 255-pas
        pas = exposure.adjust_gamma(pas, 3)
        pas = (pas > threshold_otsu(pas))
        pas = binary_closing(pas,disk(1))
        pas = pas.astype(np.uint8)

        boundary_w_mem = binary_dilation(mask,disk(20))
        blim = boundary_w_mem
        indel = binary_erosion(boundary_w_mem,disk(20))
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



        #tbmdist = distance_transform_edt(tbm)
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

    return tubs


def process_arteriol_features(mask_xml, art_value, mpp):

    arteries = mask_xml == art_value
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

    return arts
