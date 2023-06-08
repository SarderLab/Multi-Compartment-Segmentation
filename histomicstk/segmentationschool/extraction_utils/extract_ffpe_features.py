import sys, cv2
import numpy as np

# import lxml.etree as ET
# from matplotlib import path
# import matplotlib.patches as patches
# import glob
# import tifffile as ti
# import xlsxwriter
# import multiprocessing
# from scipy.ndimage.morphology import distance_transform_edt
# from scipy.ndimage import binary_fill_holes
# from skimage.transform import resize
# from skimage.util import img_as_ubyte



from skimage.morphology import binary_erosion,disk

from skimage.filters import *


# from vitessce.data_utils import optimize_arr
# from anndata import AnnData

#NOTES:
# - combine all features into single csv with 0s for features in other compartments
# - Add a column for slide name
# - Make sure feature names are distinguished from other compartments
NAMES_DICT = {'non_globally_sclerotic_glomeruli':3,
              'globally_sclerotic_glomeruli':4,
              'tubules':5,
              'arteries/arterioles':6}

def imreconstruct(marker: np.ndarray, mask: np.ndarray, radius: int = 1):
    """Iteratively expand the markers white keeping them limited by the mask during each iteration.
    :param marker: Grayscale image where initial seed is white on black background.
    :param mask: Grayscale mask where the valid area is white on black background.
    :param radius Can be increased to improve expansion speed while causing decreased isolation from nearby areas.
    :returns A copy of the last expansion.
    Written By Semnodime.
    """
    kernel = np.ones(shape=(radius * 2 + 1,) * 2, dtype=np.uint8)
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        cv2.bitwise_and(src1=expanded, src2=mask, dst=expanded)

        # Termination criterion: Expansion didn't change the image at all
        if (marker == expanded).all():
            return expanded
        marker = expanded

def xml_to_mask(annotations, location, size, downsample_factor=1, verbose=0):
    # parse xml and get root
    # tree = ET.parse(xml_path)
    # root = tree.getroot()

    # calculate region bounds
    bounds = {'x_min' : location[0], 'y_min' : location[1], 'x_max' : location[0] + size[0], 'y_max' : location[1] + size[1]}

    sorted_Annotations = sorted(annotations, key=lambda x: NAMES_DICT[(x['annotation']['name']).strip()])

    IDs = regions_in_mask(Annotations=sorted_Annotations, bounds=bounds, verbose=verbose)

    if verbose != 0:
        print('\nFOUND: ' + str(len(IDs)) + ' regions')

    # find regions in bounds
    Regions = get_vertex_points(Annotations=sorted_Annotations, verbose=verbose)

    # fill regions and create mask
    mask = Regions_to_mask(Regions=Regions, bounds=bounds, IDs=IDs, downsample_factor=downsample_factor, verbose=verbose)
    if verbose != 0:
        print('done...\n')

    return mask

def restart_line(): # for printing labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def regions_in_mask(Annotations, bounds, verbose=1):
    # find regions to save
    IDs = []

    for Annotation in Annotations: # for all annotations
        annotationName = Annotation['annotation']['name'].strip()

        annotationID = NAMES_DICT[annotationName]

        for Region in Annotation['annotation']['elements']: # iterate on all region

            if verbose != 0:
                sys.stdout.write('TESTING: ' + 'Annotation: ' + annotationID + '\tRegion: ' + Region['id'])
                sys.stdout.flush()
                restart_line()

            for Vertex in Region['points']: # iterate on all vertex in region
                # get points
                x_point = np.int32(np.float64(Vertex[0]))
                y_point = np.int32(np.float64(Vertex[1]))
                # test if points are in bounds
                if bounds['x_min'] <= x_point <= bounds['x_max'] and bounds['y_min'] <= y_point <= bounds['y_max']: # test points in region bounds
                    # save region Id
                    IDs.append({'regionID' : Region['id'], 'annotationID' : annotationID, 'annotationName':annotationName})
                    break
    return IDs

def get_vertex_points(Annotations, verbose=1):
    Regions = []
    for Annotation in Annotations: # for all annotations
        if verbose != 0:
            #sys.stdout.write('PARSING: ' + 'Annotation: ' + ID['annotationID'] + '\tRegion: ' + ID['regionID'])
            sys.stdout.flush()
            restart_line()

        # get all vertex attributes (points)
        Vertices = []
        for Region in Annotation['annotation']['elements']:
            for Vertex in Region['points']:#root.findall("./Annotation[@Id='" + ID['annotationID'] + "']/Regions/Region[@Id='" + ID['regionID'] + "']/Vertices/Vertex"):
            # make array of points
                Vertices.append([int(float(Vertex[0])), int(float(Vertex[1]))])


        Regions.append(np.array(Vertices))

    return Regions

def Regions_to_mask(Regions, bounds, IDs, downsample_factor, verbose=1):
    downsample = int(np.round(downsample_factor**(.5)))
    strel=disk(3)
    if verbose !=0:
        print('\nMAKING MASK:')

    if len(Regions) != 0: # regions present
        # get min/max sizes
        min_sizes = np.empty(shape=[2,0], dtype=np.int32)
        max_sizes = np.empty(shape=[2,0], dtype=np.int32)
        for Region in Regions: # fill all regions
            min_bounds = np.reshape((np.amin(Region, axis=0)), (2,1))
            max_bounds = np.reshape((np.amax(Region, axis=0)), (2,1))
            min_sizes = np.append(min_sizes, min_bounds, axis=1)
            max_sizes = np.append(max_sizes, max_bounds, axis=1)
        min_size = np.amin(min_sizes, axis=1)
        max_size = np.amax(max_sizes, axis=1)



        # add to old bounds
        bounds['x_min_pad'] = min(min_size[1], bounds['x_min'])
        bounds['y_min_pad'] = min(min_size[0], bounds['y_min'])
        bounds['x_max_pad'] = max(max_size[1], bounds['x_max'])
        bounds['y_max_pad'] = max(max_size[0], bounds['y_max'])

        # make blank mask
        mask = np.zeros([ int(np.round((bounds['y_max_pad'] - bounds['y_min_pad']) / downsample)), int(np.round((bounds['x_max_pad'] - bounds['x_min_pad']) / downsample)) ], dtype=np.int8)
        mask_temp = np.zeros([ int(np.round((bounds['y_max_pad'] - bounds['y_min_pad']) / downsample)), int(np.round((bounds['x_max_pad'] - bounds['x_min_pad']) / downsample)) ], dtype=np.int8)

        # fill mask polygons
        # index = 0
        for idx,Region in enumerate(Regions):

            # reformat Regions
   
            Region[:,1] = np.int32(np.round((Region[:,1] - bounds['y_min_pad']) / downsample))
            Region[:,0] = np.int32(np.round((Region[:,0] - bounds['x_min_pad']) / downsample))

            x_start = np.int32((np.round((bounds['x_min'] - bounds['x_min_pad'])) / downsample))
            y_start = np.int32((np.round((bounds['y_min'] - bounds['y_min_pad'])) / downsample))
            x_stop = np.int32((np.round((bounds['x_max'] - bounds['x_min_pad'])) / downsample))
            y_stop = np.int32((np.round((bounds['y_max'] - bounds['y_min_pad'])) / downsample))

            # get annotation ID for mask color
            ID = IDs[idx]
            '''
            if int(ID['annotationID'])==4:
                xl=x_stop-x_start
                yl=y_stop-y_start
                Region2[:,0]=Region2[:,0]-x_start
                Region2[:,1]=Region2[:,1]-y_start
                for vert in Region2:
                    if vert[0]<0:
                        vert[0]=0
                    if vert[1]<0:
                        vert[1]=0
                    if vert[0]>xl:
                        vert[0]=xl
                    if vert[1]>yl:
                        vert[1]=yl



                mask_temp = np.zeros([int((xl) / downsample),int((yl) / downsample)], dtype=np.int8)

                cv2.fillPoly(mask_temp, [Region2], int(ID['annotationID']))


                s=disk(2)
                e=binary_erosion(mask_temp,s).astype('uint8')
                d=binary_dilation(mask_temp,s).astype('uint8')
                tub_divider=np.where((d-e)==1)

                mask_temp=mask_temp.astype('uint8')
                mask_temp[tub_divider]=5

                temp_pull=mask[ y_start:y_stop, x_start:x_stop ]
                temp_pull[np.where(mask_temp==4)]=4
                temp_pull[np.where(mask_temp==5)]=1
                mask[ y_start:y_stop, x_start:x_stop ]=temp_pull
            else:
            '''


            if int(ID['annotationID'])==5:
                #print(np.float(idx)/np.float(len(Regions)))

                #t=time.time()
                cv2.fillPoly(mask_temp, [Region], int(ID['annotationID']))
                #print(time.time()-t)
                x1=np.min(Region[:,1])
                x2=np.max(Region[:,1])
                y1=np.min(Region[:,0])
                y2=np.max(Region[:,0])
                #t=time.time()
                sub_mask=mask_temp[x1:x2,y1:y2]
                #print(time.time()-t)


                #t=time.time()
                e=binary_erosion(sub_mask,strel).astype('uint8')
                #print(time.time()-t)

                #t=time.time()
                #d=binary_dilation(sub_mask,strel).astype('uint8')
                #print(time.time()-t)

                #t=time.time()
                #tub_divider=np.where((d-e)==1)
                #print(time.time()-t)

                #t=time.time()
                #sub_mask[tub_divider]=1

                #print(time.time()-t)

                #t=time.time()
                tub_prev=mask[x1:x2,y1:y2]
                tub_prev[e==1]=int(ID['annotationID'])
                #sub_mask[tub_divider]=1

                #overlap=tub_prev&sub_mask
                #sub_mask[overlap]=1
                mask[x1:x2,y1:y2]=tub_prev

                #print(time.time()-t)
            else:
                cv2.fillPoly(mask, [Region], int(ID['annotationID']))
            # index = index + 1

        # reshape mask

        # pull center mask region
        mask = mask[ y_start:y_stop, x_start:x_stop ]
        #plt.imshow(mask*50)
        #plt.show()
        '''
        msub=np.zeros((y_stop-y_start,x_stop-x_start))
        msub2=msub
        msub[np.where(mask==4)]=1
        s=disk(3)
        msub=binary_dilation(msub,selem=s)-msub2
        mask[np.where(msub==1)]=5
        '''
    else: # no Regions
        mask = np.zeros([ int(np.round((bounds['y_max'] - bounds['y_min']) / downsample)), int(np.round((bounds['x_max'] - bounds['x_min']) / downsample)) ],dtype=np.int8)

    return mask
