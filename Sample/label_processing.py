#!/usr/bin/env python
# coding: utf-8

# In[4]:


#label processing
import glob
import re

def create_label(label_file_name):
    labels={}
    
    with open(label_file_name,'r') as f:

        for line in f.readlines():

            tokens=line.split("\t")


            if any(i.isdigit() for i in tokens[0]) and any(i.isdigit() for i in tokens[1]) :              
                labels[tokens[0]]=tokens[1].strip("\n")
                    
    return labels
     

create_label('label_summary.txt')






# In[20]:


import glob
import os
import numpy as np

def load_label_for_images_in_folder(imagepath):
    
    dic=create_label("label_summary.txt")
    labels=[]
    

    
    path=imagepath+'\*jpeg'
    filenames=glob.glob(path)
    
    print(len(filenames))
    for name in filenames:
        key_value=name.rsplit('_',-1)[2].strip(".jpeg")
        if key_value in dic:
            labels.append(dic[key_value])
        else :
            os.remove(name)
            print(name+" removed.")
    return labels

load_label_for_images_in_folder("all")


# In[8]:


import tensorflow as tf
import cv2
import random
import glob
tf.enable_eager_execution()
import numpy as np

def create_image_and_label(image_path,image_size):
    datas=[]
    data_root=image_path
    dic=create_label("label_summary.txt")
    path=data_root+'\*jpeg'
    filenames=glob.glob(path)
    
    random.shuffle(filenames)
    labels=[]
                    
    for name in filenames:
        key_value=name.rsplit('_',-1)[2].strip(".jpeg")
        if key_value in dic:
            labels.append((dic[key_value]))
            datas.append(cv2.imread(name,cv2.IMREAD_GRAYSCALE))
                    
    datas=np.array(datas).reshape(-1,image_size,image_size,1)
    return datas,labels

get_ipython().run_line_magic('time', 'datas,labels=create_image_and_label("all",224)')


# In[9]:


#trying out pickle to save datas and labels
import pickle


pickle_out=open("datas.pickle","wb")
pickle.dump(datas,pickle_out)

pickle_out.close()

pickle_out=open("labels.pickle","wb")
pickle.dump(labels,pickle_out)

pickle_out.close()



pickle_in=open("datas.pickle","rb")
X=pickle.load(pickle_in)

