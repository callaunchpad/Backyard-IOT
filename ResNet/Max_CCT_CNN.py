import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import json
import os
import cv2
import numpy as np
import random
import os

"""Prepare Data"""

images_path = "eccv_18_all_images_sm"
labels_path = "CaltechCameraTrapsECCV18.json"

with open(labels_path) as json_file:
    data = json.load(json_file)

clist, cdict = [], {}
for category in data['categories']:
    cdict[category['id']] = len(clist)
    clist.append(category['name'])

num_categories = len(clist)

adict = {}
for annotation in data['annotations']:
    adict[annotation['image_id']] = cdict[annotation['category_id']]

amount = 1000
dim = (1000, 500)
path_lst = next(os.walk(images_path))[2]
image_lst, label_lst = [], []
i = 1
for idx in random.sample(range(len(path_lst)), amount):
    image = path_lst[idx]
    
    imarr = cv2.imread(images_path + '/' + image, cv2.IMREAD_COLOR)
    imarr = cv2.resize(imarr, dim, interpolation = cv2.INTER_AREA)
    
    image_lst.append(imarr)
    
    label_vec = np.zeros(num_categories)
    label_vec[adict[image.split('.')[0]]] = 1
    label_lst.append(label_vec)
    
    if i % 50 == 0:
        print(i / amount * 100, '% done')
    i+=1
    
image_arr, label_arr = np.array(image_lst), np.array(label_lst)

path = 'data2/'

os.mkdir(path)

np.save(path + 'images.npy', image_arr)
np.save(path + 'labels.npy', label_arr)
np.savez(path + 'label_to_animal.npz', clist)