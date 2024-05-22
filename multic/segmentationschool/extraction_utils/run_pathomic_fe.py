import sys
import os, girder_client
import numpy as np
from tqdm import tqdm
import pandas as pd
from tiffslide import TiffSlide
import lxml.etree as ET
import multiprocessing
from joblib import Parallel, delayed

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

from .extract_ffpe_features import xml_to_mask

from .process_mc_features import process_glom_features, process_tubules_features, process_arteriol_features
from segmentationschool.Codes.xml_to_mask_minmax import write_minmax_to_xml
from segmentationschool.utils.json_to_xml import get_xml_path

def getPathomicFeatures(args):

    cores=multiprocessing.cpu_count()
    folder = args.base_dir

    # assert args.target is not None, 'Directory of xmls must be specified, use --target /path/to/files.xml'
    # assert args.wsis is not None, 'Directory of WSIs must be specified, use --wsis /path/to/wsis'
    if args.platform == 'DSA':
        import girder_client
        gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
        gc.setToken(args.girderToken)

        file_name = args.file.split('/')[-1]
        slide_item_id = args.item_id
        output_dir = args.base_dir
        slide_name,slideExt=file_name.split('.')
        items=[(args.file, args.xml_path)]

    elif args.platform == 'HPG':
        image_files = [image_name for _, _, files in os.walk(folder) for image_name in files if image_name.endswith('.xml')]
        image_names = [os.path.join(folder, f.split('.')[0]) for f in image_files]
        slideExt = args.ext
        output_dir = args.output_dir
        # each item in items is a tuple of (images, annotations)
        items = [(f + slideExt, f + '.xml') for f in image_names]
    else:
        raise Exception("Please Enter a valid Platform, DSA or HPG")

    for i in range(len(items)):

        svsfile, xmlfile = items[i]

        print(xmlfile,'here')
        write_minmax_to_xml(xmlfile)
        slide = TiffSlide(svsfile)

        all_contours = {'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}
        tree = ET.parse(xmlfile)
        root = tree.getroot()

        xlsx_path = os.path.join(output_dir, os.path.basename(svsfile).split('.')[0] +'_pathomic'+'.xlsx')
        for Annotation in root.findall("./Annotation"): # for all annotations
            annotationID = Annotation.attrib['Id']
            if annotationID not in ['1','2','3','4','5','6']:
                pass
            else:
                for Region in Annotation.findall("./*/Region"): # iterate on all region
                    verts=[]
                    for Vert in Region.findall("./Vertices/Vertex"): # iterate on all vertex in region
                        verts.append([int(float(Vert.attrib['X'])),int(float(Vert.attrib['Y']))])
                    all_contours[annotationID].append(np.array(verts))

        glom_features=Parallel(n_jobs=cores,prefer="threads")(delayed(process_glom_features)(points,
            MOD, slide, h_threshold=args.h_threshold, saturation_threshold=args.saturation_threshold) for points in tqdm(all_contours['3'],colour='yellow',unit='Glomerulus',leave=False))
        s_glom_features=Parallel(n_jobs=cores,prefer="threads")(delayed(process_glom_features)(points,
                MOD, slide, h_threshold=args.h_threshold, saturation_threshold=args.saturation_threshold) for points in tqdm(all_contours['4'],colour='yellow',unit='Scl. Glomerulus',leave=False))
        tub_features = Parallel(n_jobs=cores,prefer="threads")(delayed(process_tubules_features)(points,
                MOD, slide,whitespace_threshold=args.whitespace_threshold) for points in tqdm(all_contours['5'],colour='blue',unit='Tubule',leave=False))
        art_features=Parallel(n_jobs=cores,prefer="threads")(delayed(process_arteriol_features)(points,
                ) for points in tqdm(all_contours['6'], colour='magenta',unit='Artery(-iole)',leave=False))
        glom_features=[i for i in glom_features if i is not None]
        s_glom_features=[i for i in s_glom_features if i is not None]
        tub_features=[i for i in tub_features if i is not None]
        art_features=[i for i in art_features if i is not None]

        def get_feature_matrix(features, columns):
            m, n = len(features), len(columns)
            feature_matrix = np.zeros((m, n)) 
            for i, feat in enumerate(features):
                for j in range(n):
                    feature_matrix[i,j] = feat[j]
                
            return feature_matrix
        # gloms = np.zeros((len(glom_features),7))
        # s_gloms = np.zeros((len(s_glom_features),7))
        # tubs = np.zeros((len(tub_features),7))
        # arts = np.zeros((len(art_features),5))

        # for i, glom in enumerate(glom_features):
        #     gloms[i,0] = glom[0]
        #     gloms[i,1] = glom[1]
        #     gloms[i,2] = glom[2]
        #     gloms[i,3] = glom[3]
        #     gloms[i,4] = glom[4]
        #     gloms[i,5] = glom[5]
        #     gloms[i,6] = glom[6]

        # for i, glom in enumerate(s_glom_features):
        #     s_gloms[i,0] = s_glom[0]
        #     s_gloms[i,1] = s_glom[1]
        #     s_gloms[i,2] = s_glom[2]
        #     s_gloms[i,3] = s_glom[3]
        #     s_gloms[i,4] = s_glom[4]
        #     s_gloms[i,5] = s_glom[5]
        #     s_gloms[i,6] = s_glom[6]

        # for i, tub in enumerate(tub_features):
        #     if not tub:
        #         continue
        #     tubs[i,0] = tub[0]
        #     tubs[i,1] = tub[1]
        #     tubs[i,2] = tub[2]
        #     tubs[i,3] = tub[3]
        #     tubs[i,4] = tub[4]
        #     tubs[i,5] = tub[5]
        #     tubs[i,6] = tub[6]


        # for i, art in enumerate(art_features):
        #     if not art:
        #         continue
        #     arts[i,0] = art[0]
        #     arts[i,1] = art[1]
        #     arts[i,2] = art[2]
        #     arts[i,3] = art[3]
        #     arts[i,4] = art[4]

    
    
        all_columns = [['x1','x2','y1','y2','Area','Mesangial Area','Mesangial Fraction'],
                    ['x1','x2','y1','y2','Area','Mesangial Area','Mesangial Fraction'],
                    ['x1','x2','y1','y2','Average TBM Thickness','Average Cell Thickness','Luminal Fraction'],
                    ['x1','x2','y1','y2','Arterial Area']]
        compart_names = ['gloms','s_gloms','tubs','arts']

        gloms = get_feature_matrix(glom_features, all_columns[0])
        s_gloms = get_feature_matrix(s_glom_features,all_columns[1])
        tubs = get_feature_matrix(tub_features, all_columns[2])
        arts = get_feature_matrix(art_features,all_columns[3])
    
        all_comparts = [gloms,s_gloms,tubs, arts]
        
        _ = os.system("printf '\tWriting Excel file: [{}]\n'".format(xlsx_path))
        with pd.ExcelWriter(xlsx_path) as writer:
            for idx,compart in enumerate(all_comparts):
                df = pd.DataFrame(compart,columns=all_columns[idx])
                df.to_excel(writer, index=False, sheet_name=compart_names[idx])
        if args.platform == 'DSA':
            gc.uploadFileToItem(file_id, xlsx_path, reference=None, mimeType=None, filename=None, progressCallback=None)
            print('Girder file uploaded!')



