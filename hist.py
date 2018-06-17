# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 17:17:42 2018

@author: 212704260
"""

import numpy as np
import skimage
from skimage import io
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os

path_to_image_dir='C:\\Users\\212704260\\Downloads\\satellite_images\\sat_img'
dirs_list= os.listdir(path_to_image_dir)
print(dirs_list)


img_hist1=[]
img_hist2=[]
img_hist3=[]
img_hist4=[]
img_hist5=[]
img_hist6=[]
img_hist7=[]
img_hist8=[]
img_hist9=[]

for dirs in dirs_list:
    for file in os.listdir(os.path.join(path_to_image_dir,dirs)):
        if file.endswith('.TIF'):
            img=io.imread(os.path.join(path_to_image_dir,dirs,file))
            hist,edges = np.histogram(img[img!=0], bins=32)
            norm_hist=sklearn.preprocessing.minmax_scale(hist)
            #norm_hist=hist/float(hist.sum())
            norm_hist=np.expand_dims(norm_hist.T,axis=1)
            
            if file.endswith('B1.TIF'):
                img_hist1.append(norm_hist)
                
            if file.endswith('B2.TIF'):
                img_hist2.append(norm_hist)
            if file.endswith('B3.TIF'):
                img_hist3.append(norm_hist)
                #img_hist3=np.concatenate((img_hist3,norm_hist),axis=1)
            if file.endswith('B4.TIF'):
                img_hist4.append(norm_hist)
                #img_hist4=np.concatenate((img_hist4,norm_hist),axis=1)
            if file.endswith('B5.TIF'):
                img_hist5.append(norm_hist)
                #img_hist5=np.concatenate((img_hist5,norm_hist),axis=1)
            if file.endswith('B6_VCID_1.TIF'):
                img_hist6.append(norm_hist)
                #img_hist6=np.concatenate((img_hist6,norm_hist),axis=1)
            if file.endswith('B7.TIF'):
                img_hist7.append(norm_hist)
                #img_hist7=np.concatenate((img_hist7,norm_hist),axis=1)
            if file.endswith('B8.TIF'):
                img_hist8.append(norm_hist)
                #img_hist8=np.concatenate((img_hist8,norm_hist),axis=1)
            if file.endswith('B6_VCID_2.TIF'):
                img_hist9.append(norm_hist)
                #img_hist9=np.concatenate((img_hist9,norm_hist),axis=1)

img_hist1=np.column_stack(tuple(img_hist1))
img_hist2=np.column_stack(tuple(img_hist2))
img_hist3=np.column_stack(tuple(img_hist3))
img_hist4=np.column_stack(tuple(img_hist4))
img_hist5=np.column_stack(tuple(img_hist5))
img_hist6=np.column_stack(tuple(img_hist6))
img_hist7=np.column_stack(tuple(img_hist7))
img_hist8=np.column_stack(tuple(img_hist8))
img_hist9=np.column_stack(tuple(img_hist9))

#hist_3d=np.empty((32,5,9))
hist_3d=np.stack((img_hist1,img_hist2,img_hist3,img_hist4,img_hist5,img_hist6,img_hist9,img_hist7,img_hist8),axis=2)       
print(hist_3d.shape)
np.save('C:\\Users\\212704260\\Downloads\\satellite_images\\sat_img\\hist.npy',hist_3d)
plt.imshow(hist_3d[:,:,4],cmap='gray')
plt.show()
      