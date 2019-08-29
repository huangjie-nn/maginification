#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
     


def create_reference(label_file_name):
    
    references={}
    
    with open(label_file_name,'r') as f:

        for line in f.readlines():

            tokens=line.split("\t")


            if any(i.isdigit() for i in tokens[0]) and any(i.isdigit() for i in tokens[1]) :              
                references[tokens[0]]=tokens[0].strip("\n")
                    
    return references




# In[3]:


def load_label_for_images_in_folder(imagepath):
    
    dic=create_label("label_summary.txt")
    labels=[]
    

    
    path=imagepath+'\*jpeg'
    filenames=glob.glob(path)
    
    for name in filenames:
        key_value=name.rsplit('_')[-1].strip(".jpeg")

        if key_value in dic:
            labels.append(dic[key_value])
        else :
            labels.append("")
            
    return labels

print(len(load_label_for_images_in_folder("all")))


# In[2]:


import cv2
import random
import glob
import numpy as np

def create_image_and_label(image_path,image_size):
    datas=[]
    data_root=image_path
    dic=create_label("label_summary.txt")
    references=create_reference("label_summary.txt")
    path=data_root+'\*jpeg'
    filenames=glob.glob(path)
    
    random.shuffle(filenames)
    labels=[]
    refs=[]
                    
    for name in filenames:
        key_value=name.rsplit('_')[-1].strip(".jpeg")
        if key_value in dic:
            labels.append((dic[key_value]))
            datas.append(cv2.imread(name,cv2.IMREAD_GRAYSCALE))
            refs.append((references[key_value]))       
    datas=np.array(datas).reshape(-1,image_size,image_size,1)
    return datas,labels,refs

get_ipython().run_line_magic('time', 'datas,labels,refs=create_image_and_label("all",32)')


# In[3]:


#trying out pickle to save datas and labels
import pickle


pickle_out=open("datas.pickle","wb")
pickle.dump(datas,pickle_out)

pickle_out.close()

pickle_out=open("labels.pickle","wb")
pickle.dump(labels,pickle_out)

pickle_out.close()

pickle_out=open("refs.pickle","wb")
pickle.dump(refs,pickle_out)

pickle_out.close()





# In[20]:


import matplotlib.pyplot as plt

pickle_in=open("datas.pickle","rb")
X=pickle.load(pickle_in)

pickle_in=open("labels.pickle","rb")
y=pickle.load(pickle_in)

print(X[0].shape)
print(len(y))


print(X[0])

