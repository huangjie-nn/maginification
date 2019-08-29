#!/usr/bin/env python
# coding: utf-8

# In[19]:



import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import errno
from os import rename, listdir
import shutil

def directory_checking(oldpath,newpath):
    if not os.path.exists(oldpath):
        raise OSError(42, 'no such file',oldpath)
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def resize_and_crop_image(path,factor,newpath):
    
    oriimg=cv2.imread(path,cv2.IMREAD_COLOR)

    
    height, width, depth = oriimg.shape
    imgScale = factor/height
    newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
    newimg = cv2.resize(oriimg,(int(newX),int(newY)))


    newW=int((newimg.shape[1])/2)
    newH=int((newimg.shape[0])/2)

    square_image=newimg[:,newW-newH:newW+newH]
    cv2.imwrite(newpath,square_image)   
    


def recursive_resizing(oldpath,newpath,factor=224):
    directory_checking(oldpath,newpath)  
    old_path=oldpath+'\*jpeg'
    filenames=glob.glob(old_path)
    for f in filenames:
        new_path=(newpath+(f).strip(oldpath))
        resize_and_crop_image(f,factor,new_path)
    print("Resizing and cropping done.")




        

def apply_gaussian_noise(oldpath, newpath, mean, var):
        
    image=cv2.imread(oldpath)

    output=image.astype("float32")

    row,col,ch= output.shape
    
    gauss = np.random.normal(mean,var**0.5,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    img_after=output/255

    image = img_after + gauss  

    noisy=(image * 255).astype(np.uint8)
    cv2.imwrite(newpath,noisy)

    return noisy



def recursive_gaussian(oldpath,newpath,mean=0,var=0.01):
    
    directory_checking(oldpath,newpath)
    
  
    old_path=oldpath+'\*jpeg'
    filenames=glob.glob(old_path)

    
    for f in filenames:

        new_path=(newpath+(f).strip(oldpath))

        apply_gaussian_noise(f,new_path,mean,var)
    
    print("Variance now: ",var)

def gaussian_for_range(start,end,step,oldpath,newpath):

    print(start,end,step)
    
    for i in np.arange(start,end,step):
        print(i)
        recursive_gaussian(oldpath,newpath+'_'+str(i),0,i/1000)
        
        
    print("Apply gaussian noises done.") 



def apply_vignetting(oldpath,newpath,var):
    img = cv2.imread(oldpath,cv2.IMREAD_COLOR)
    
    rows,cols,channel = img.shape

    a = cv2.getGaussianKernel(cols,var)
    b = cv2.getGaussianKernel(rows,var)
    c = b*a.T

    
    
    mask = 255 * c / np.linalg.norm(c)
    output = np.copy(img)


    for i in range(channel):
        output[:,:,i] = output[:,:,i] * mask

    cv2.imwrite(newpath,output)   

def recursive_vignetting(oldpath,newpath,var):
    
    directory_checking(oldpath,newpath)

  
    old_path=oldpath+'\*jpeg'
    filenames=glob.glob(old_path)

    
    for f in filenames:

        new_path=(newpath+(f.replace(oldpath,'')))
        
        apply_vignetting(f,new_path,var)

    




def vigneeting_for_range(begin,end,step,oldpath):

    for i in np.arange(begin,end,step):
        recursive_vignetting(oldpath,oldpath+'_vignetting'+'_'+str(i),i)
    
    



def apply_vigneeting_bunch(start,end,step,oldpath):

    for file in os.listdir('.'):
        filename = os.fsdecode(file)
        if filename.startswith("gaus"): 

            print(filename)
            vigneeting_for_range(start,end,step,filename)

            print('Vigneeting for '+filename+" done")
    
    print("Apply vigneeting done.") 
    



def change_image_name_in_folder(oldpath,newpath):
    
    old_path=oldpath+'\*jpeg'
    filenames=glob.glob(old_path)
    for name in filenames:

        ori=name.split('\\',1)[-1]
        new_path=newpath+'\\'+oldpath+'_'+ori

        
        
        rename(name,new_path)

        
def change_back_image_name(oldpath):
    old_path=oldpath+'\*jpeg'
    filenames=glob.glob(old_path)
    for name in filenames:

        ori=name.split('_',3)[-1]
        new_path=oldpath+'\\'+ori
        print(name)
        print(ori)
        print(new_path)

        
        rename(name,new_path)
    
        


   

def copy_file_to_antoher_directory(oldpath,newpath):

    directory_checking(oldpath,newpath)

    old_path=oldpath+'\*jpeg'

    filenames=glob.glob(old_path)

    for name in filenames:

        ori=name.split('\\',1)[-1]
        new_path=newpath+'\\'+oldpath+'_'+ori
       
        shutil.copy(name,new_path)
        
        
def change_image_name_in_batch(keyword,newpath):

    for file in os.listdir('.'):
        filename = os.fsdecode(file)
        if filename.startswith(keyword):
            copy_file_to_antoher_directory(filename,newpath)


def zoom_mag_data_processing_pipeline(oldpath,newpath):
    recursive_resizing(oldpath,'resize')
    gaussian_for_range(5,51,5,'resize','gaus')
    os.rename("resize","gaus_0")
    apply_vigneeting_bunch(50,301,50,'gaus')
    change_image_name_in_batch("gaus_",newpath)






# In[20]:


change_image_name_in_batch("gaus_","all")

