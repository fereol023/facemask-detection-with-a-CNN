# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:22:38 2022

@author: gbeno
"""
import pickle
import sklearn
from sklearn.model_selection import train_test_split
#from scipy import misc
import matplotlib.image as mpimg
import numpy as np
import os

label = os.listdir("facemask_dataset/Train")
label = label[1:]
dataset = []

for image_label in label:
    images = os.listdir("facemask_dataset/Train/"+image_label)
   
    for image in images:
        img = mpimg.imread("facemask_dataset/Train/"+image_label+"/"+image)
        #img = misc.imresize(img, (64, 64))
        dataset.append((img,image_label))
X = []
Y = []

for input_,image_label in dataset:
    X.append(input_)
    Y.append(label.index(image_label))

X = np.array(X)
Y = np.array(Y)

X_train,y_train, = X,Y

data_set = (X_train,y_train)

save_label = open("int_to_word_out.pickle","wb")
pickle.dump(label, save_label)
save_label.close()

if __name__ == '__main__':
    
    print(data_set)