import numpy as np
from tqdm import tqdm
from skimage.color import rgb2lab, rgb2hsv
from skimage import exposure
from skimage.morphology import remove_small_objects,binary_erosion,binary_dilation,disk,binary_opening,binary_closing
from skimage.filters import threshold_otsu
from skimage import measure
import cv2
from .extract_ffpe_features import imreconstruct
from .PAS_deconvolution import deconvolution


def process_glom_features(points, MOD, slide, h_threshold, saturation_threshold):
    
    #Area, Cellularity, Mesangial Area to Cellularity Ratio
    area = cv2.contourArea(points)
    if area>0:
        y1, y2, x1, x2 =[np.min(points[:,0]),np.max(points[:,0]),np.min(points[:,1]),np.max(points[:,1])]

        crop = slide.read_region((y1,x1),0,(y2-y1,x2-x1))
        crop = np.array(crop)
        crop = crop[:,:,:3]

        mask = np.zeros((x2-x1,y2-y1),dtype=np.uint8)
        points[:,0]-=y1
        points[:,1]-=x1
        mask=cv2.fillPoly(mask,[points],1)

        hsv = rgb2hsv(crop)
        hsv = hsv[:,:,1]
        pas_seg = hsv > saturation_threshold
        pas_seg = pas_seg.astype(np.uint8)
        pas_seg = np.multiply(pas_seg,mask)
        pas_seg = pas_seg.astype(np.uint8)

        h,_,_ = deconvolution(crop,MOD)

        h = 255-h
        # h = (h>threshold_otsu(h))
        h = h>h_threshold
        h = h.astype(np.uint8)


        pas_seg = ((pas_seg - h) > 0).astype(np.uint8)

        mask_pixels = np.sum(mask)
        pas_pixels = np.sum(pas_seg)

        mes_fraction = (pas_pixels) / area

        return [x1,x2,y1,y2, area, pas_pixels, mes_fraction]

def process_tubules_features(points, MOD, slide, whitespace_threshold):

    area = cv2.contourArea(points)
    if area>0:
        y1, y2, x1, x2 =[np.min(points[:,0]),np.max(points[:,0]),np.min(points[:,1]),np.max(points[:,1])]

        crop = slide.read_region((y1,x1),0,(y2-y1,x2-x1))
        crop = np.array(crop)
        crop = crop[:,:,:3]

        mask = np.zeros((x2-x1,y2-y1),dtype=np.uint8)
        points[:,0]-=y1
        points[:,1]-=x1
        mask=cv2.fillPoly(mask,[points],1)

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
        
        return [x1,x2,y1,y2,tbm_avg,cyto_avg,np.sum(WS) / np.sum(mask)]


def process_arteriol_features(points):

    area = cv2.contourArea(points)
    if area>0:
        y1, y2, x1, x2 =[np.min(points[:,0]),np.max(points[:,0]),np.min(points[:,1]),np.max(points[:,1])]
        return [x1,x2,y1,y2,area]
