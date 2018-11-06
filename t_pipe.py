from __future__ import division

import math
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import Iterator
from keras.utils.np_utils import to_categorical
import keras.backend as K

from t_utils import rotate
from t_crop import my_math,trim_contours,trim_iterative,trim_contours_exact,trim_recursive
from t_writ import save_set_image

def pipeline(model,input,size=None,preprocess_func=None,start_angle=0,stop_angle=180,step_angle=2,m_dir=None):
    # if isinstance(input, (np.ndarray)):
    #     images = input
    #     N, h_o, w_o = images.shape[:3]
    #     if not size:
    #         size = (h_o, w_o)
    #     indexes = np.random.choice(N, num_images)
    #     images = images[indexes, ...]
    # else:
    images = []
    filenames = input
    N = len(filenames)
    #indexes = np.random.choice(N, num_images)
    for i in range(N):
        image = cv2.imread(filenames[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    images = np.asarray(images)
    
    x = []
    r_img = []
    c_img = []
    for image in images:
        #q.append(image)
        for t in range(start_angle,stop_angle+step_angle,step_angle):
            img_r=rotate(image,t)
            r_img.append(img_r)
            image_s = cv2.resize(img_r,size)
            x.append(image_s)
    save_set_image(r_img,"r",step_angle,m_dir)
    
    x = np.asarray(x, dtype='float32')
    if x.ndim == 3:
        x = np.expand_dims(x, axis=3)
    if preprocess_func:
        x = preprocess_func(x)
    y_pred = np.argmax(model.predict(x), axis=1)



    for rotated_image, predicted_angle in zip(r_img,y_pred):
        #r_w,r_l,no_f=rotated_image.shape
        corrected_image = rotate(rotated_image, -predicted_angle)
        
        #f_img=my_math(corrected_image,r_w,r_l,predicted_angle,err)
        f_img=trim_recursive(corrected_image)
        #f_img=trim_contours(corrected_image)
        #f_img=trim_contours_exact(corrected_image)
        #f_img=trim_iterative(corrected_image)
        c_img.append(f_img)
    save_set_image(c_img,"c",step_angle,m_dir)


          