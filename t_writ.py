from __future__ import division

import math
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import Iterator
from keras.utils.np_utils import to_categorical
import keras.backend as K

def save_image(r,c,i,di):

    f_name="rotated"+str(i)+".jpg"
    os.chdir(di+"/Output/Rotated")
    r=cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f_name,r)

    f_name="correct"+str(i)+".jpg"
    os.chdir(di+"/Output/Corrected")
    c=cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f_name,c)
    
    os.chdir(di)

def save_set_image(q,type,step_angle,di):

    for i in range(len(q)):
        f_name=type+str(i*step_angle)+".jpg"
        os.chdir(di+"/Output/"+type)
        r=cv2.cvtColor(q[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(f_name,r)
    
    os.chdir(di)