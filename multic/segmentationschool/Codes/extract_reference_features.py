import os, cv2
import numpy as np

import lxml.etree as ET
import girder_client
from matplotlib import path
from skimage.color import rgb2lab,rgb2hsv

from .xml_to_mask_minmax import write_minmax_to_xml
import xlsxwriter
import multiprocessing
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import binary_fill_holes

from tqdm import tqdm
import time
from tiffslide import TiffSlide
from joblib import Parallel, delayed
from skimage.color import rgb2hsv

from skimage.filters import *

def getKidneyReferenceFeatures(args):

    # assert args.target is not None, 'Directory of xmls must be specified, use --target /path/to/files.xml'
    # assert args.wsis is not None, 'Directory of WSIs must be specified, use --wsis /path/to/wsis'

    gc = girder_client.GirderClient(apiUrl=args.girderApiUrl)
    gc.setToken(args.girderToken)

    folder = args.base_dir
    girder_folder_id = folder.split('/')[-2]
    file_name = args.files.split('/')[-1]
    files = list(gc.listItem(girder_folder_id))
    item_dict = dict()

    for file in files:
        d = {file['name']: file['_id']}
        item_dict.update(d)

    slide_item_id = item_dict[file_name]

    #output_dir = args.base_dir + '/tmp'
    slide_name,slideExt=file_name.split('.')
    xlsx_path = slide_name + '.xlsx'

    annotatedXMLs=[args.xml_path]
    for xml in annotatedXMLs:
        # print(xml,end='\r',flush=True)

        print(xml,'here')
        write_minmax_to_xml(xml)
        # for ext in args.wsi_ext.split(','):

        #     if os.path.isfile(os.path.join(args.wsis,xml.split('/')[-1].replace('.xml',ext))):
        #         wsi=os.path.join(args.wsis,xml.split('/')[-1].replace('.xml',ext))
        #         break
        slideExt=file_name.split('.')[-1]
        all_contours = {'1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}
        # cortex medulla glomeruli scl_glomeruli tubules arteries(ioles)
        tree = ET.parse(xml)
        root = tree.getroot()
        basename=os.path.splitext(xml)[0]
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

        cortexcontour=[]
        cortexcodes=[]
        medullacontour=[]
        medullacodes=[]
        cortexarea=0
        medullaarea=0

        slide=TiffSlide(args.files)
        if slideExt =='.scn':
            dim_x=int(slide.properties['tiffslide.bounds-width'])## add to columns
            dim_y=int(slide.properties['tiffslide.bounds-height'])## add to rows
            offsetx=int(slide.properties['tiffslide.bounds-x'])##start column
            offsety=int(slide.properties['tiffslide.bounds-y'])##start row

        elif slideExt in ['.ndpi','.svs','.tiff']:
            dim_x, dim_y=slide.dimensions
            offsetx=0
            offsety=0


        fullSize=slide.level_dimensions[0]
        resRatio= args.chop_thumbnail_resolution
        ds_1=fullSize[0]/resRatio
        ds_2=fullSize[1]/resRatio

        thumbIm=np.array(slide.get_thumbnail((ds_1,ds_2)))
        slide.close()
        if slideExt =='.scn':
            xStt=int(offsetx/resRatio)
            xStp=int((offsetx+dim_x)/resRatio)
            yStt=int(offsety/resRatio)
            yStp=int((offsety+dim_y)/resRatio)
            thumbIm=thumbIm[yStt:yStp,xStt:xStp]

        hsv=rgb2hsv(thumbIm)

        binary=(hsv[:,:,1]>0.1).astype('bool')

        black=hsv[:,:,2]<.5

        binary[black]=0
        binary=binary_fill_holes(binary)

        total_tissue_area=np.sum(binary)*resRatio*resRatio
  
        for contour in all_contours['1']:
            cortexcontour.extend(contour)
            cortexarea+=cv2.contourArea(contour)
            cortexcodes.extend([path.Path.MOVETO])
            for i in range(1,np.shape(contour)[0]-1):
                cortexcodes.extend([path.Path.LINETO])
            cortexcodes.extend([path.Path.CLOSEPOLY])
        if len(cortexcontour)>0:
            cortex_path=path.Path(cortexcontour,codes=cortexcodes)
        else:
            cortex_path=None

        for contour in all_contours['2']:
            medullacontour.extend(contour)
            medullacodes.extend([path.Path.MOVETO])
            medullaarea+=cv2.contourArea(contour)
            for i in range(1,np.shape(contour)[0]-1):
                medullacodes.extend([path.Path.LINETO])
            medullacodes.extend([path.Path.CLOSEPOLY])
        if len(medullacontour)>0:
            medulla_path=path.Path(medullacontour,codes=medullacodes)
        else:
            medulla_path=None
        pseudocortexarea=total_tissue_area-medullaarea
        #
        workbook=xlsxwriter.Workbook(xlsx_path)
        worksheet1 = workbook.add_worksheet('Summary')
        worksheet2 = workbook.add_worksheet('Interstitium')
        worksheet3 = workbook.add_worksheet('Glomeruli')
        worksheet4 = workbook.add_worksheet('Sclerosed glomeruli')
        worksheet5 = workbook.add_worksheet('Tubules')
        worksheet6 = workbook.add_worksheet('Arteries - Arterioles')
        cores=multiprocessing.cpu_count()
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
        glom_features=Parallel(n_jobs=cores)(delayed(points_to_features_glom)(points,
            args,args.min_size[2],cortex_path,medulla_path) for points in tqdm(all_contours['3'],colour='yellow',unit='Glomerulus',leave=False))
        sglom_features=Parallel(n_jobs=cores)(delayed(points_to_features_glom)(points,
            args,args.min_size[3],cortex_path,medulla_path) for points in tqdm(all_contours['4'],colour='red',unit='Scl. glomerulus',leave=False))
        tub_features=Parallel(n_jobs=cores)(delayed(points_to_features_tub)(points,
            args,args.min_size[4],cortex_path,medulla_path) for points in tqdm(all_contours['5'],colour='blue',unit='Tubule',leave=False))

        art_features=Parallel(n_jobs=cores)(delayed(points_to_features_art)(points,
            args,args.min_size[5],cortex_path,medulla_path,args.files,MOD) for points in tqdm(all_contours['6'],colour='magenta',unit='Artery(-iole)',leave=False))
        print('Generating output file..')
        glom_features=[i for i in glom_features if i is not None]
        sglom_features=[i for i in sglom_features if i is not None]
        tub_features=[i for i in tub_features if i is not None]
        art_features=[i for i in art_features if i is not None]

        # gloms_features=[i for i in glom_features if i[0]>args.min_sizes[2]]
        # sglom_features=[i for i in sglom_features if i[0]>args.min_sizes[3]]

        # cortexgloms=[i for i in glom_features if not i[3]]
        cortextubs=[i for i in tub_features if not i[3]]
        cortexarts=[i for i in art_features if not i[3]]

        medullatubs=[i for i in tub_features if i[3]]
        medullaarts=[i for i in art_features if i[3]]


        if pseudocortexarea>0:
            cortex_glom_area=np.sum(np.array(glom_features)[:,0])
            cortex_glom_density=float(cortex_glom_area)/float(pseudocortexarea)
            cortex_tub_area=np.sum(np.array(cortextubs)[:,0])
            cortex_tub_density=float(cortex_tub_area)/float(pseudocortexarea)
            cortex_art_area=np.sum(np.array(cortexarts)[:,0])
            cortex_art_density=float(cortex_art_area)/float(pseudocortexarea)
            # downsample_cortex=get_downsample_cortex(args,all_contours['1'])
            # exit()
            # capillary_densities=Parallel(n_jobs=cores)(delayed(get_capillary_densities)(points,
            #     args,args.min_size[4],cortex_path,medulla_path,wsi) for points in tqdm(all_contours['6'],colour='magenta',unit='Artery(-iole)'))
        else:
            cortex_glom_density=None
            cortex_tub_density=None
            cortex_art_density=None
        if medullaarea>0 and len(medullatubs)>0:

            medulla_tub_area=np.sum(np.array(medullatubs)[:,0])
            if len(medullaarts)>0:
                medulla_art_area=np.sum(np.array(medullaarts)[:,0])
                medulla_art_density=float(medulla_art_area)/float(medullaarea)
            else:
                medulla_art_density=None
            medulla_tub_density=float(medulla_tub_area)/float(medullaarea)
        else:
            medulla_tub_density=None
            medulla_art_density=None
        glom_features=np.array(glom_features)
        sglom_features=np.array(sglom_features)
        tub_features=np.array(tub_features)
        art_features=np.array(art_features)
        cortexarts=np.array(cortexarts)
        worksheet1.write(0,0,'Glomerular density - count:')
        worksheet1.write(0,1,len(glom_features)/pseudocortexarea)
        worksheet1.write(1,0,'Average glomerular area:')
        worksheet1.write(1,1,np.mean(glom_features[:,0]))
        worksheet1.write(2,0,'Std glomerular area:')
        worksheet1.write(2,1,np.std(glom_features[:,0]))
        worksheet1.write(3,0,'Average glomerular radius:')
        worksheet1.write(3,1,np.mean(glom_features[:,1]))
        worksheet1.write(4,0,'Std glomerular radius:')
        worksheet1.write(4,1,np.std(glom_features[:,1]))
        worksheet1.write(5,0,'Glomerulosclerosis density - count')
        worksheet1.write(6,0,'Average scl.glomerular area:')
        worksheet1.write(7,0,'Std scl.glomerular area:')
        worksheet1.write(8,0,'Average scl.glomerular radius:')
        worksheet1.write(9,0,'Std scl.glomerular radius:')
        if len(sglom_features)>0:
            worksheet1.write(5,1,len(sglom_features)/pseudocortexarea)
            worksheet1.write(6,1,np.mean(sglom_features[:,0]))
            worksheet1.write(7,1,np.std(sglom_features[:,0]))
            worksheet1.write(8,1,np.mean(sglom_features[:,1]))
            worksheet1.write(9,1,np.std(sglom_features[:,1]))

        worksheet1.write(10,0,'Cortical tubular density')

        worksheet1.write(11,0,'Average cortical tubular area:')
        worksheet1.write(12,0,'Std cortical tubular area:')
        worksheet1.write(13,0,'Average cortical tubular radius:')
        worksheet1.write(14,0,'Std cortical tubular radius:')

        worksheet1.write(15,0,'Average medullary tubular area:')
        worksheet1.write(16,0,'Std medullary tubular area:')
        worksheet1.write(17,0,'Average medullary tubular radius:')
        worksheet1.write(18,0,'Std medullary tubular radius:')
        if pseudocortexarea>0:
            cortextubs=np.array(cortextubs)
            worksheet1.write(10,1,len(cortextubs)/pseudocortexarea)

            worksheet1.write(11,1,np.mean(cortextubs[:,0]))
            worksheet1.write(12,1,np.std(cortextubs[:,0]))
            worksheet1.write(13,1,np.mean(cortextubs[:,1]))
            worksheet1.write(14,1,np.std(cortextubs[:,1]))
        if medullaarea>0:
            medullatubs=np.array(medullatubs)
            worksheet1.write(15,1,np.mean(medullatubs[:,0]))
            worksheet1.write(16,1,np.std(medullatubs[:,0]))
            worksheet1.write(17,1,np.mean(medullatubs[:,1]))
            worksheet1.write(18,1,np.std(medullatubs[:,1]))

        worksheet1.write(19,0,'Cortical arterial(olar) density')
        worksheet1.write(19,1,len(cortexarts)/pseudocortexarea)
        worksheet1.write(20,0,'Average lumen to wall ratio:')
        ltwr=np.array([i for i in art_features[:,4] if i is not None])
        worksheet1.write(20,1,np.mean(ltwr))

        worksheet1.write(21,0,'Glomerulosclerosis ratio:')
        worksheet1.write(21,1,float(len(sglom_features))/float(len(sglom_features)+len(glom_features)))
        worksheet1.write(22,0,'Cortical glomerular density - area:')
        worksheet1.write(22,1,cortex_glom_density)
        worksheet1.write(23,0,'Cortical tubular density - area')
        worksheet1.write(23,1,cortex_tub_density)
        worksheet1.write(24,0,'Cortical artery(iole) density - area')
        worksheet1.write(24,1,cortex_art_density)
        worksheet1.write(25,0,'Medullary tubular density - area')
        worksheet1.write(25,1,medulla_tub_density)

        worksheet1.write(26,0,'Medullary arteriole density - area')
        worksheet1.write(26,1,medulla_art_density)
        worksheet1.write(27,0,'Gloms/cortex tubules ratio:')
        worksheet1.write(27,1,np.sum(glom_features[:,0])/np.sum(cortextubs[:,0]))
        cInterstitial_area=cortexarea-np.sum(cortextubs[:,0])-np.sum(cortexarts[:,0])-np.sum(glom_features[:,0])
        if len(medullatubs)>0:
            mInterstitial_area=medullaarea-np.sum(medullatubs[:,0])
        else:
            mInterstitial_area=0
        if len(sglom_features)>0:
            cInterstitial_area-=np.sum(sglom_features[:,0])
        worksheet1.write(28,0,'Cortical interstitial density')
        worksheet1.write(28,1,cInterstitial_area/pseudocortexarea)
        worksheet1.write(29,0,'Overall tubule density - count')
        worksheet1.write(29,1,len(tub_features)/total_tissue_area)
        worksheet1.write(29,0,'Total nephron density - count')
        worksheet1.write(29,1,(len(tub_features)+len(sglom_features)+len(glom_features))/total_tissue_area)
        worksheet1.write(30,0,'Medullary interstitial density')
        if medullaarea>0:
            worksheet1.write(30,1,mInterstitial_area/medullaarea)
        worksheet1.write(31,0,'Cortical tubule count')
        worksheet1.write(31,1,len(cortextubs))
        worksheet1.write(32,0,'Cortical artery/arteriole count')
        worksheet1.write(32,1,len(cortexarts))
        worksheet1.write(33,0,'Glomerulus count')
        worksheet1.write(33,1,len(glom_features))
        worksheet1.write(34,0,'sGlomerulus count')
        worksheet1.write(34,1,len(sglom_features))
        worksheet3.write(0,0,'Area')
        worksheet3.write(0,1,'Radius')
        for idx,glom in enumerate(glom_features):
            worksheet3.write(idx+1,0,glom[0])
            worksheet3.write(idx+1,1,glom[1])

        worksheet4.write(0,0,'Area')
        worksheet4.write(0,1,'Radius')
        for idx,sglom in enumerate(sglom_features):
            worksheet4.write(idx+1,0,sglom[0])
            worksheet4.write(idx+1,1,sglom[1])

        worksheet5.write(0,0,'Area')
        worksheet5.write(0,1,'Radius')
        for idx,tub in enumerate(tub_features):
            worksheet5.write(idx+1,0,tub[0])
            worksheet5.write(idx+1,1,tub[1])

        worksheet5.write(0,2,'Cortical areas')
        worksheet5.write(0,3,'Cortical radii')
        for idx,tub in enumerate(cortextubs):
            worksheet5.write(idx+1,2,tub[0])
            worksheet5.write(idx+1,3,tub[1])

        worksheet5.write(0,4,'Medullary areas')
        worksheet5.write(0,5,'Medullary radii')
        for idx,tub in enumerate(medullatubs):
            worksheet5.write(idx+1,4,tub[0])
            worksheet5.write(idx+1,5,tub[1])

        worksheet6.write(0,0,'Area')
        worksheet6.write(0,1,'Radius')
        worksheet6.write(0,2,'Luminal ratio')
        for idx,art in enumerate(art_features):
            worksheet6.write(idx+1,0,art[0])
            worksheet6.write(idx+1,1,art[1])
            worksheet6.write(idx+1,2,art[4])


        try:
            workbook.close(save_to=xlsx_path)
        except:
            print("An exception occurred")

        gc.uploadFileToItem(slide_item_id, xlsx_path, reference=None, mimeType=None, filename=None, progressCallback=None)
        print('Done.')
        # exit()
    # Parallel(n_jobs=num_cores, backend='threading')(delayed(chop_wsi)(, choppable_regions=choppable_regions)  for idxx, j in enumerate(index_x))


# def summarizeKidneyReferenceFeatures(args):
#     #assert args.target is not None, 'Directory of xmls must be specified, use --target /path/to/files.xml'
#     assert args.SummaryOption is not None, 'You must specify what type of summary is required with --SummaryOption'
#     assert args.patientData is not None, 'You must provide patient metadata xlsx file with --patientData'
#     if args.SummaryOption in ['ULDensity']:
#         ULDensity(args)
#     elif args.SummaryOption in ['BLDensity']:
#         BLDensity(args)
#     elif args.SummaryOption in ['standardScatter']:
#         standardScatter(args)
#     elif args.SummaryOption in ['anchorScatter']:
#         anchorScatter(args)
#     elif args.SummaryOption in ['aggregate']:
#         aggregate(args)
#     elif args.SummaryOption in ['JoyPlot']:
#         JoyPlot(args)
#     else:
#         print('Incorrect SummaryOption')

# def anchorScatter(args):
#     patientData=pd.read_excel(args.patientData,usecols=[args.labelColumns,args.IDColumn,args.anchor],index_col=None)
#     patientMetrics={}
#     for idx,patientID in enumerate(patientData[args.IDColumn]):
#         patientMetrics[patientID]=[patientData[args.labelColumns][idx],patientData[args.anchor][idx]]

#     datafiles=glob.glob(os.path.join(args.target, "*.xlsx"))
#     clinical_legend=[]
#     clinical_anchor=[]
#     for datafile in datafiles:
#         patientID=os.path.splitext(datafile.split('/')[-1])[0]
#         clinical_legend.append(str(patientMetrics[patientID][0]))
#         clinical_anchor.append(patientMetrics[patientID][1])

#     sortedPatientOrder=[x for  _, x in sorted(zip(clinical_legend,datafiles))]
#     sortedPatientLegend=[x for x,_ in sorted(zip(clinical_legend,datafiles))]
#     sortedPatientAnchor=[x for x,_ in sorted(zip(clinical_anchor,datafiles))]
#     f1,f2=[int(i) for i in args.scatterFeatures.split(',')]
#     # f1=5
#     # f2=7

#     temp=pd.read_excel(datafiles[0],sheet_name='Summary',header=None,index_row=None,index_col=0)
#     index = temp.index
#     xlabel=index[f1-1]
#     ylabel=args.anchor

#     popIdx=[]
#     scatter_features=[]
#     for idx,datafile in enumerate(tqdm(sortedPatientOrder,colour='red')):
#         features=np.array(pd.read_excel(datafile,sheet_name='Summary',header=None,index_row=None,index_col=0))
#         # print(np.array(features))
#         if not np.isnan(features[f1-1]) and not np.isnan(features[f2-1]):
#             scatter_features.append([features[f1-1][0],features[f2-1][0]])
#         else:
#             popIdx.append(idx)
#     for p in popIdx[::-1]:
#         sortedPatientLegend.pop(p)
#         sortedPatientAnchor.pop(p)
#     scatter_features=np.array(scatter_features)

#     sns.scatterplot(x=scatter_features[:,0],y=sortedPatientAnchor,hue=sortedPatientLegend,palette='viridis',legend='auto')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend(title=args.labelColumns)
#     plt.show()
# def standardScatter(args):
#     patientData=pd.read_excel(args.patientData,usecols=[args.labelColumns,args.IDColumn],index_col=None,converters={args.IDColumn:str})
#     patientMetrics={}
#     assert args.labelColumns in ['Age','Cr','Sex'], 'Label column must be Age, Cr, or Sex for standard scatter'
#     if args.labelColumns=='Age':
#         labelBins=[0,10,20,30,40,50,60,70,80,90,100]
#     elif args.labelColumns=='Cr':
#         labelBins=[0,.5,.6,.7,.8,.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
#     elif args.labelColumns=='Sex':
#         labelBins=None

#     for idx,patientID in enumerate(patientData[args.IDColumn]):
#         patientMetrics[patientID]=patientData[args.labelColumns][idx]

#     datafiles=glob.glob(os.path.join(args.target, "*.xlsx"))
#     clinical_legend=[]
#     for datafile in datafiles:
#         patientID=os.path.splitext(datafile.split('/')[-1])[0]
#         clinical_legend.append(str(patientMetrics[patientID]))

#     sortedPatientOrder=[x for  _, x in sorted(zip(clinical_legend,datafiles))]
#     sortedPatientLegend=[x for x,_ in sorted(zip(clinical_legend,datafiles))]
#     print(sortedPatientLegend)
#     f1,f2=[int(i) for i in args.scatterFeatures.split(',')]
#     # f1=5
#     # f2=7

#     temp=pd.read_excel(datafiles[0],sheet_name='Summary',header=None,index_row=None,index_col=0)
#     index = temp.index
#     xlabel=index[f1-1]
#     ylabel=index[f2-1]

#     popIdx=[]
#     scatter_features=[]
#     for idx,datafile in enumerate(tqdm(sortedPatientOrder,colour='red')):
#         features=np.array(pd.read_excel(datafile,sheet_name='Summary',header=None,index_row=None,index_col=0))
#         # print(np.array(features))
#         if not np.isnan(features[f1-1]) and not np.isnan(features[f2-1]):
#             scatter_features.append([features[f1-1][0],features[f2-1][0]])
#         else:
#             popIdx.append(idx)
#     for p in popIdx[::-1]:
#         sortedPatientLegend.pop(p)
#     sortedPatientLegend=np.array(sortedPatientLegend)

#     scatter_features=np.array(scatter_features)

#     sns.scatterplot(x=scatter_features[:,0],y=scatter_features[:,1],hue=sortedPatientLegend,palette='viridis')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend(sortedPatientLegend,title=args.labelColumns)
#     plt.show()

# def BLDensity(args):
#     patientData=pd.read_excel(args.patientData,usecols=[args.labelColumns,args.IDColumn],index_col=None)
#     patientMetrics={}
#     for idx,patientID in enumerate(patientData[args.IDColumn]):
#         patientMetrics[patientID]=patientData[args.labelColumns][idx]

#     datafiles=glob.glob(os.path.join(args.target, "*.xlsx"))
#     clinical_legend=[]
#     for datafile in datafiles:
#         patientID=os.path.splitext(datafile.split('/')[-1])[0]
#         clinical_legend.append(str(patientMetrics[patientID]))

#     sortedPatientOrder=[x for  _, x in sorted(zip(clinical_legend,datafiles))]
#     sortedPatientLegend=[x for x,_ in sorted(zip(clinical_legend,datafiles))]

#     fig,ax=plt.subplots()
#     plt.xlabel('Cortical tubular diameter (µm)')
#     plasma = cm.get_cmap('plasma',len(sortedPatientOrder))
#     popIdx=[]
#     for idx,datafile in enumerate(tqdm(sortedPatientOrder)):
#         tubule_features=pd.read_excel(datafile,sheet_name='Tubules',usecols=['Cortical radii','Cortical areas'],index_col=None).dropna()
#         if len(tubule_features['Cortical radii'])>0:
#             # xl1=np.percentile(tubule_features['Cortical radii']*0.25,0.01)
#             xl2=np.percentile(tubule_features['Cortical radii']*0.25,99)
#             # yl1=np.percentile(tubule_features['Cortical areas']*0.25*0.25,0.01)
#             yl2=np.percentile(tubule_features['Cortical areas']*0.25*0.25,99)
#             # sns.kdeplot(tubule_features['Cortical radii']*0.25,ax=ax,clip=[l1,l2],bw_adjust=0.75,color=plasma(idx),fill=args.plotFill)
#             sns.kdeplot(x=tubule_features['Cortical radii']*0.25,y=tubule_features['Cortical areas']*0.25*0.25,
#                 color=plasma(idx),ax=ax,clip=[[None,xl2],[None,yl2]])
#         else:
#             popIdx.append(idx)
#     for p in popIdx[::-1]:
#         sortedPatientLegend.pop(p)
#     plt.legend(sortedPatientLegend,title=args.labelColumns)
#     plt.show()

# def ULDensity(args):

#     if args.labelModality=='Continuous':
#         patientData=pd.read_excel(args.patientData,usecols=[args.labelColumns,args.IDColumn],index_col=None,converters={args.IDColumn:str})
#         patientMetrics={}
#         # print(patientData)
#         for idx,patientID in enumerate(patientData[args.IDColumn]):
#             patientMetrics[patientID]=patientData[args.labelColumns][idx]

#         datafiles=glob.glob(os.path.join(args.target, "*.xlsx"))
#         clinical_legend=[]
#         for datafile in datafiles:
#             patientID=os.path.splitext(datafile.split('/')[-1])[0]
#             clinical_legend.append(str(patientMetrics[patientID]))

#         sortedPatientOrder=[x for  _, x in sorted(zip(clinical_legend,datafiles))]
#         sortedPatientLegend=[x for x,_ in sorted(zip(clinical_legend,datafiles))]

#         fig,ax=plt.subplots()
#         plt.xlabel('Cortical tubular diameter (µm)')
#         plasma = cm.get_cmap('plasma',len(sortedPatientOrder))
#         popIdx=[]
#         all_tubules=[]
#         featurename='Cortical radii'
#         sheetname='Tubules'
#         for idx,datafile in enumerate(tqdm(sortedPatientOrder,colour='red')):
#             tubule_features=pd.read_excel(datafile,sheet_name=sheetname,usecols=[featurename],index_col=None).dropna()
#             if len(tubule_features[featurename])>0:
#                 l1=np.percentile(tubule_features[featurename]*0.25,0.05)
#                 l2=np.percentile(tubule_features[featurename]*0.25,99.5)
#                 sns.kdeplot(tubule_features[featurename]*0.25,ax=ax,clip=[l1,l2],bw_adjust=0.75,color=plasma(idx),fill=args.plotFill,alpha=0.1)
#                 # sns.kdeplot(tubule_features[featurename]*0.25,ax=ax,bw_adjust=0.75,color=plasma(idx),fill=args.plotFill,alpha=0.1)
#                 all_tubules.extend(np.array(tubule_features[featurename]*0.25))
#             else:
#                 popIdx.append(idx)
#         for p in popIdx[::-1]:
#             sortedPatientLegend.pop(p)

#         plt.legend(sortedPatientLegend,title=args.labelColumns)
#         plt.title('Cumulative distributions per patient')
#         # plt.colorbar()
#         plt.show()
#         # print(len(all_tubules))
#         # fig,ax=plt.subplots()
#         # plt.xlabel('Cortical tubular diameter (µm)')
#         # sns.kdeplot(all_tubules,ax=ax,bw_adjust=0.75,fill=args.plotFill)
#         # plt.title('Cumulative distribution for dataset')
#         # plt.show()
#     elif args.labelModality=='Categorical':
#         patientData=pd.read_excel(args.patientData,usecols=[args.labelColumns,args.IDColumn],index_col=None,converters={args.IDColumn:str})
#         patientMetrics={}
#         for idx,patientID in enumerate(patientData[args.IDColumn]):
#             patientMetrics[patientID]=patientData[args.labelColumns][idx]

#         datafiles=glob.glob(os.path.join(args.target, "*.xlsx"))
#         clinical_legend=[]
#         for datafile in datafiles:
#             patientID=os.path.splitext(datafile.split('/')[-1])[0]

#             clinical_legend.append(str(patientMetrics[patientID]))

#         fig,ax=plt.subplots()
#         plt.xlabel('Cortical tubular diameter (µm)')
#         plasma = cm.get_cmap('plasma',len(np.unique(clinical_legend))+1)
#         labelMapper={}
#         categories=np.unique(clinical_legend)
#         for idx,l in enumerate(categories):
#             labelMapper[l]=idx
#         popIdx=[]
#         all_tubules=[]
#         featurename='Radius'
#         sheetname='Glomeruli'
#         for idx,datafile in enumerate(tqdm(datafiles,colour='red')):
#             tubule_features=pd.read_excel(datafile,sheet_name=sheetname,usecols=[featurename],index_col=None).dropna()
#             if len(tubule_features[featurename])>0:
#                 l1=np.percentile(tubule_features[featurename]*0.25,0.01)
#                 l2=np.percentile(tubule_features[featurename]*0.25,99.9)

#                 pcolor=plasma(labelMapper[clinical_legend[idx]])
#                 # sns.kdeplot(tubule_features[featurename]*0.25,ax=ax,clip=[l1,l2],bw_adjust=0.75,color=pcolor,fill=args.plotFill,alpha=0.5)
#                 sns.kdeplot(tubule_features[featurename]*0.25,ax=ax,bw_adjust=0.75,color=pcolor,fill=args.plotFill,alpha=0.5)

#                 all_tubules.extend(np.array(tubule_features[featurename]*0.25))
#             else:
#                 popIdx.append(idx)
#         for p in popIdx[::-1]:
#             clinical_legend.pop(p)
#         plt.legend(clinical_legend,title=args.labelColumns)
#         plt.title('Cumulative distributions per patient')
#         plt.show()
#         # fig,ax=plt.subplots()
#         # plt.xlabel('Cortical tubular diameter (µm)')
#         # sns.kdeplot(all_tubules,ax=ax,bw_adjust=0.75,fill=args.plotFill)
#         # plt.title('Cumulative distribution for dataset')
#         # plt.show()


# def JoyPlot(args):

#     url = "https://gist.githubusercontent.com/borgar/31c1e476b8e92a11d7e9/raw/0fae97dab6830ecee185a63c1cee0008f6778ff6/pulsar.csv"
#     df = pd.read_csv(url, header=None)
#     df = df.stack().reset_index()
#     df.columns = ['idx', 'x', 'y']

#     patientData=pd.read_excel(args.patientData,usecols=[args.labelColumns,args.IDColumn],index_col=None,converters={args.IDColumn:str})
#     patientMetrics={}
#     for idx,patientID in enumerate(patientData[args.IDColumn]):
#         patientMetrics[patientID]=patientData[args.labelColumns][idx]
#     datafiles=glob.glob(os.path.join(args.target, "*.xlsx"))
#     clinical_legend=[]
#     for datafile in datafiles:
#         patientID=os.path.splitext(datafile.split('/')[-1])[0]
#         clinical_legend.append(str(patientMetrics[patientID]))

#     sortedPatientOrder=[x for  _, x in sorted(zip(clinical_legend,datafiles))]
#     sortedPatientLegend=[x for x,_ in sorted(zip(clinical_legend,datafiles))]

#     popIdx=[]
#     all_tubules=[]
#     # outDF=pd.DataFrame()

#     for idx,datafile in enumerate(tqdm(sortedPatientOrder[:15],colour='red')):

#         tubule_features=pd.read_excel(datafile,sheet_name='Tubules',usecols=['Cortical radii'],index_col=None).dropna()
#         if len(tubule_features['Cortical radii'])>0:
#             #.values()
#             feature_array=np.array(tubule_features['Cortical radii']*0.25)
#             all_tubules.append(feature_array)

#             l1=np.percentile(feature_array,0.05)
#             l2=np.percentile(feature_array,99.5)
#             # sns.kdeplot(tubule_features['Cortical radii']*0.25,ax=ax,clip=[l1,l2],bw_adjust=0.75,color=plasma(idx),fill=args.plotFill,cbar=True)
#             # all_tubules.extend(np.array(tubule_features['Cortical radii']*0.25))

#         else:
#             popIdx.append(idx)
#     for p in popIdx[::-1]:
#         sortedPatientLegend.pop(p)
#     outDF=pd.DataFrame(all_tubules)

#     outDF=outDF.stack(dropna=False).reset_index().dropna()

#     # outDFT=outDF.transpose()
#     # input(outDF)
#     # input(outDF.reset_index())
#     # input(outDF.reset_index(drop=True)[::2].reset_index(drop=True))
#     # input(outDF.reset_index(drop=True))
#     # input(outDF.stack())
#     # input(outDF.reset_index().stack())
#     # input(outDF.reset_index(drop=True).stack())
#     # outDF=outDF.reset_index(drop=True)[::2].reset_index(drop=True)
#     # outDF = outDF.stack().reset_index()
#     outDF.columns = ['idx', 'x', 'y']
#     outDF.to_csv('test.csv')
#     # print(outDF.dropna())
#     # exit()




#     sns.set_theme(rc={"axes.facecolor": (0, 0, 0,0), 'figure.facecolor':'#ffffff', 'axes.grid':False})
#     g = sns.FacetGrid(outDF, row='idx',hue="idx", aspect=15)
#     g.map(sns.kdeplot, "x",
#           bw_adjust=.9, clip_on=False,
#           fill=True, alpha=1, linewidth=1.5)
#     g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.9)
#     g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
#     def label(x, color, label):
#         ax = plt.gca()
#         ax.text(0, .2, label, fontweight="bold", color=color,
#                 ha="left", va="center", transform=ax.transAxes)
#     g.map(label, "x")

#     # Draw the densities in a few steps
#     # g.map(sns.lineplot, 'x', 'y', clip_on=False, alpha=1, linewidth=1.5)
#     # g.map(plt.fill_between, 'x', 'y', color='#000000')
#     # g.map(sns.lineplot, 'x', 'y', clip_on=False, color='#ffffff', lw=2)
#     # Set the subplots to overlap
#     g.fig.subplots_adjust(hspace=-0.95)
#     g.set_titles("")
#     g.set(yticks=[], xticks=[], ylabel="", xlabel="")
#     g.despine(bottom=True, left=True)
#     plt.show()
#     # plt.savefig('joy.png', facecolor='#000000')

# def aggregate(args):
#     assert args.exceloutfile is not None, 'You must provide a name of xlsx output file for feature aggregation with --exceloutfile name.xlsx'
#     usecols=args.labelColumns.split(',')
#     # index_col=int(args.IDColumn)
#     patientData=pd.read_excel(args.patientData,usecols=usecols.extend(args.IDColumn),index_col=args.IDColumn)
#     patientData = patientData.loc[:, ~patientData.columns.str.contains('^Unnamed')]
#     patientMetrics={}

#     pd.set_option('display.max_rows', 100)
#     patientData.index = patientData.index.map(str)

#     datafiles=glob.glob(os.path.join(args.target, "*.xlsx"))
#     # numGloms=0
#     # numSGloms=0
#     # numTubules=0
#     # numArts=0
#     full_data={}
#     for idx,datafile in enumerate(tqdm(datafiles,colour='red')):

#         xlsxid=datafile.split('/')[-1].split('.xlsx')[0]
#         tubule_features=pd.read_excel(datafile,sheet_name='Summary',header=None,index_col=None).transpose()
#         names=np.array(tubule_features.iloc[0])
#         vals=np.array(tubule_features.iloc[1])
#         # numGloms+=vals[0]
#         # if not np.isnan(vals[5]):
#         #     numSGloms+=vals[5]
#         # numTubules+=vals[10]
#         # numArts+=vals[19]

#         full_data[xlsxid]=np.append(vals,np.array(patientData.loc[xlsxid]),0)
#     # print(numGloms)
#     # print(numSGloms)
#     # print(numTubules)
#     # print(numArts)
#     # exit()
#     workbook=xlsxwriter.Workbook(args.exceloutfile,{'nan_inf_to_errors': True})
#     worksheet1 = workbook.add_worksheet('Aggregation')

#     outcolnames=np.append(names,patientData.columns,0)
#     outrownames=full_data.keys()
#     for idx,outcol in enumerate(outcolnames):
#         worksheet1.write(0,idx+1,outcol)

#     rowcounter=1
#     for key,vals in full_data.items():
#         worksheet1.write(rowcounter,0,key)
#         for idx,val in enumerate(vals):
#             worksheet1.write(rowcounter,idx+1,val)
#         rowcounter+=1

#     workbook.close()
def points_to_features_glom(points,args,min_size,cortex,medulla):
    a=cv2.contourArea(points)
    if a>min_size:
        # if cortex is not None:
        #     containedcortex=any(cortex.contains_points(points))
        # else:
        #     containedcortex=False
        if medulla is not None:
            containedmedulla=any(medulla.contains_points(points))
        else:
            containedmedulla=False
        xMin,xMax,yMin,yMax=[np.min(points[:,0]),np.max(points[:,0]),np.min(points[:,1]),np.max(points[:,1])]
        binary_mask=np.zeros((yMax-yMin,xMax-xMin))
        points[:,0]-=xMin
        points[:,1]-=yMin
        binary_mask=cv2.fillPoly(binary_mask,[points],1)
        dist=distance_transform_edt(binary_mask)


        return [a,np.max(dist),None,containedmedulla]

def points_to_features_tub(points,args,min_size,cortex,medulla):
    a=cv2.contourArea(points)
    if a>min_size:
        # if cortex is not None:
        #     containedcortex=any(cortex.contains_points(points))
        # else:
        #     containedcortex=False
        if medulla is not None:
            containedmedulla=any(medulla.contains_points(points))
        else:
            containedmedulla=False
        xMin,xMax,yMin,yMax=[np.min(points[:,0]),np.max(points[:,0]),np.min(points[:,1]),np.max(points[:,1])]
        binary_mask=np.zeros((yMax-yMin,xMax-xMin))
        points[:,0]-=xMin
        points[:,1]-=yMin
        binary_mask=cv2.fillPoly(binary_mask,[points],1)
        dist=distance_transform_edt(binary_mask)

        return [a,np.max(dist),None,containedmedulla]


def points_to_features_art(points,args,min_size,cortex,medulla,wsi,MOD):
    a=cv2.contourArea(points)
    resizeFlag=0
    if a>min_size:
        # if cortex is not None:
        #     containedcortex=any(cortex.contains_points(points))
        # else:
        #     containedcortex=False
        if medulla is not None:
            containedmedulla=any(medulla.contains_points(points))
        else:
            containedmedulla=False
        xMin,xMax,yMin,yMax=[np.min(points[:,0]),np.max(points[:,0]),np.min(points[:,1]),np.max(points[:,1])]
        image=np.array(TiffSlide(wsi).read_region((xMin,yMin),0,(xMax-xMin,yMax-yMin)))[:,:,:3]

        if xMax-xMin>5000 or yMax-yMin>5000:
            return [a,None,None,containedmedulla,None]
        binary_mask=np.zeros((int(yMax-yMin),int(xMax-xMin)))
        points[:,0]-=xMin
        points[:,1]-=yMin
        binary_mask=cv2.fillPoly(binary_mask,[points],1)

        LAB=rgb2lab(image)
        # HSV=rgb2hsv(image)

        LAB[:,:,0]/=100

        # stain1,stain2,stain3=deconvolution(image,MOD)
        #prevent divide by zero errors by adding very small nonzero
        # stain2=np.invert(stain2)/255
        # stain3=np.invert(stain3)
        # WSsatscaling=np.divide(LAB[:,:,0],HSV[:,:,1]+0.000000001)
        # WSsatscaling2=np.divide(LAB[:,:,0],stain2+0.000000001)
        # WSseg=WSsatscaling2>3
        # WSseg=LAB[:,:,0]>.65
        # fig,ax=try_all_threshold(LAB[:,:,0],figsize=(10,8))
        # plt.show()
        # WSseg=LAB[:,:,0]>threshold_lcoal(LAB[:,:,0],tolerance=2)

        dist=distance_transform_edt(binary_mask)
        # localThresh=rank.otsu(LAB[:,:,0],disk(5))
        distMax=np.max(dist)
        # print(distMax)
        # tim=threshold_local(grayImage,block_size=3,method='mean')
        WSseg=LAB[:,:,0]>.7

        # WSseg=binary_fill_holes(WSseg)

        # WSseg=binary_dilation(WSseg,disk(2))
        WSseg[WSseg!=binary_mask.astype('bool')]=0
        WSseg=binary_fill_holes(WSseg)
        # WSseg=binary_opening(WSseg,disk(1))
        # WSseg=binary_erosion(WSseg,disk(2))

        WSdist=distance_transform_edt(WSseg)

        # plt.subplot(221)
        # plt.imshow(image)
        # plt.subplot(222)
        # plt.imshow(WSseg)
        # plt.subplot(223)
        # plt.imshow(grayImage)
        # plt.subplot(224)
        # plt.imshow(image)
        # plt.title(np.max(WSdist)/np.max(dist))
        # plt.show()
        # if resizeFlag:
        #
        #     return [a,distMax*2,containedcortex,containedmedulla,(np.max(WSdist)*2)/(distMax*2)]
        # else:
        return [a,distMax,None,containedmedulla,np.max(WSdist)/distMax]



# #     full_list.append(tubule_features)
# # full_list= pd.concat(full_list)
# # print(full_list)
# # exit()
