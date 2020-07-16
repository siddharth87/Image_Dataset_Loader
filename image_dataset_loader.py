# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 23:54:45 2020

@author: 91701
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import shuffle_arrays_unison
import random

def load_dataset(path):
    filenames=path
    cs=[]
    ca=os.listdir(filenames)
    count=-1
    x_train=list()
    for c in ca:
        count=count+1
        p=os.path.join(filenames,c)
        fi=os.listdir(p)
        for f in fi:
            path1=os.path.join(p,f)
            image=cv2.imread(path1)
            image=cv2.resize(image,(64,64))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb=image_rgb/255
            x_train.append(image_rgb)
            cs.append(count)
    x_train=np.array(x_train)
    y_train=[cs]
    y_train=np.array(y_train)
    y_train=y_train.T

    shuffler = np.random.permutation(len(x_train))
    x = x_train[shuffler]
    y = y_train[shuffler]

    x=x.flatten().reshape(x.shape[1]*x.shape[2]*x.shape[3],x.shape[0])
    y=y.T




