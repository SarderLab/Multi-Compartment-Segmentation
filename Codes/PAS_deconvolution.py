import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.stats import zscore
from skimage.exposure import adjust_gamma
from skimage.filters import threshold_local

def deconvolution(img,MOD):
    MODx=MOD[0]
    MODy=MOD[1]
    MODz=MOD[2]
    cosx=np.zeros((3,))
    cosy=np.zeros((3,))
    cosz=np.zeros((3,))
    len=np.zeros((3,))
    for i in range(0,3):
        cosx[i]=cosy[i]=cosz[i]=0.0
        len[i]=np.sqrt(MODx[i]*MODx[i] + MODy[i]*MODy[i] + MODz[i]*MODz[i])
        if len[i]!=0.0:
            cosx[i]= MODx[i]/len[i]
            cosy[i]= MODy[i]/len[i]
            cosz[i]= MODz[i]/len[i]

    if cosx[1]==0.0:
        if cosy[1]==0.0:
            if cosz[1]==0.0:
                cosx[1]=cosz[0]
                cosy[1]=cosx[0]
                cosz[1]=cosy[0]

    if cosx[2]==0.0:
    	if cosy[2]==0.0:
    		if cosz[2]==0.0:
    			if ((cosx[0]*cosx[0] + cosx[1]*cosx[1])> 1):
    				cosx[2]=0.0
    			else:
    				cosx[2]=np.sqrt(1.0-(cosx[0]*cosx[0])-(cosx[1]*cosx[1]))

    			if ((cosy[0]*cosy[0] + cosy[1]*cosy[1])> 1):
    				cosy[2]=0.0
    			else:
    				cosy[2]=np.sqrt(1.0-(cosy[0]*cosy[0])-(cosy[1]*cosy[1]))

    			if ((cosz[0]*cosz[0] + cosz[1]*cosz[1])> 1):
    				cosz[2]=0.0
    			else:
    				cosz[2]=np.sqrt(1.0-(cosz[0]*cosz[0])-(cosz[1]*cosz[1]))
    leng= np.sqrt(cosx[2]*cosx[2] + cosy[2]*cosy[2] + cosz[2]*cosz[2])

    cosx[2]= cosx[2]/leng
    cosy[2]= cosy[2]/leng
    cosz[2]= cosz[2]/leng


    A = cosy[1] - cosx[1] * cosy[0] / cosx[0]
    V = cosz[1] - cosx[1] * cosz[0] / cosx[0]
    C = cosz[2] - cosy[2] * V/A + cosx[2] * (V/A * cosy[0] / cosx[0] - cosz[0] / cosx[0])
    q=np.zeros((9,))
    q[2] = (-cosx[2] / cosx[0] - cosx[2] / A * cosx[1] / cosx[0] * cosy[0] / cosx[0] + cosy[2] / A * cosx[1] / cosx[0]) / C;
    q[1] = -q[2] * V / A - cosx[1] / (cosx[0] * A);
    q[0] = 1.0 / cosx[0] - q[1] * cosy[0] / cosx[0] - q[2] * cosz[0] / cosx[0];
    q[5] = (-cosy[2] / A + cosx[2] / A * cosy[0] / cosx[0]) / C;
    q[4] = -q[5] * V / A + 1.0 / A;
    q[3] = -q[4] * cosy[0] / cosx[0] - q[5] * cosz[0] / cosx[0];
    q[8] = 1.0 / C;
    q[7] = -q[8] * V / A;
    q[6] = -q[7] * cosy[0] / cosx[0] - q[8] * cosz[0] / cosx[0];

    img_stain1 = np.ravel(np.copy(img[:,:,0]))
    img_stain2 = np.ravel(np.copy(img[:,:,1]))
    img_stain3 = np.ravel(np.copy(img[:,:,2]))
    dims=img.shape
    imagesize = dims[0] * dims[1]
    rvec=np.ravel(np.copy(img[:,:,0])).astype('float')
    gvec=np.ravel(np.copy(img[:,:,1])).astype('float')
    bvec=np.ravel(np.copy(img[:,:,2])).astype('float')
    log255=np.log(255.0)
    for i in range(0,imagesize):
        R = rvec[i]
        G = gvec[i]
        B = bvec[i]

        Rlog = -((255.0*np.log((R+1)/255.0))/log255)
        Glog = -((255.0*np.log((G+1)/255.0))/log255)
        Blog = -((255.0*np.log((B+1)/255.0))/log255)
        for j in range(0,3):
            Rscaled = Rlog * q[j*3];
            Gscaled = Glog * q[j*3+1];
            Bscaled = Blog * q[j*3+2];

            output = np.exp(-((Rscaled + Gscaled + Bscaled) - 255.0) * log255 / 255.0)
            if(output>255):
                output=255

            if j==0:
                img_stain1[i] = np.floor(output+.5)
            elif j==1:
                img_stain2[i] = np.floor(output+.5)
            else:
            	img_stain3[i] = np.floor(output+.5)
    img_stain1=np.reshape(img_stain1,(dims[0],dims[1]))
    img_stain2=np.reshape(img_stain2,(dims[0],dims[1]))
    img_stain3=np.reshape(img_stain3,(dims[0],dims[1]))
    return img_stain1,img_stain2,img_stain3
#
# img=np.array(imread('test.jpg')).astype('uint8')
#
# MODx=np.zeros((3,))
# MODy=np.zeros((3,))
# MODz=np.zeros((3,))
# MODx[0]= 0.644211
# MODy[0]= 0.716556
# MODz[0]= 0.266844
#
# MODx[1]= 0.175411
# MODy[1]= 0.972178
# MODz[1]= 0.154589
#
# MODx[2]= 0.0
# MODy[2]= 0.0
# MODz[2]= 0.0
# MOD=[MODx,MODy,MODz]
# stain1,stain2,stain3=deconvolution(img,MOD)
# MODx[2]= 1
# MODy[2]= 1
# MODz[2]= 1
# MOD=[MODx,MODy,MODz]
# stain4,stain5,stain6=deconvolution(img,MOD)
#
# plt.subplot(231)
# plt.imshow(np.invert(stain1))
# plt.subplot(232)
# plt.imshow(np.invert(stain2))
# plt.subplot(233)
# plt.imshow(np.invert(stain3))
# plt.subplot(234)
# plt.imshow(np.invert(stain4))
# plt.subplot(235)
# plt.imshow(np.invert(stain5))
# plt.subplot(236)
# plt.imshow(np.invert(stain6))
# plt.show()
#
# sat=np.invert(stain5);
# sat=adjust_gamma(sat,2)
# mems=sat>threshold_local(sat,block_size=151,offset=-0.05)
#
# plt.imshow(mems)
# plt.show()
# stainRatio=np.invert(stain1)/np.invert(stain2)
# stainRatio=zscore(stainRatio)
# plt.imshow(stainRatio)
# plt.show()
