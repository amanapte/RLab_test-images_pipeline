from __future__ import division

import math
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import Iterator
from keras.utils.np_utils import to_categorical
import keras.backend as K

def my_math(corrected_image,r_w,r_l,predicted_angle,err):
    theta=(predicted_angle*math.pi)/180
    sin=math.sin(theta)
    cos=math.cos(theta)
    A = np.array([[sin, cos], [cos, sin]])
    B = np.array([[r_l], [r_w]])
    lit=(np.linalg.inv(A) @ B)
    [w,l]=lit.flatten()
    #w=int(w)
    #l=int(l)    
    s_y=int(err*(l*(math.sin(2*theta))/2))
    e_y=int((w+l*math.sin(2*theta))-(err*(l*math.sin(2*theta)/2)))
    s_x=int(err*(w*(math.sin(2*theta))/2))
    e_x=int((l+w*math.sin(2*theta))-(err*(w*math.sin(2*theta)/2)))
    #print(s_y)
    #print(e_y)
    #print(s_x)
    #print(e_x)
    f_img=corrected_image[s_y:e_y,s_x:e_x]
    return f_img

def trim_recursive(frame):
    if frame.shape[0] == 0:
        return np.zeros((0,0,3))

    # crop top
    if not np.sum(frame[0]):
        return trim_recursive(frame[1:])
    # crop bottom
    elif not np.sum(frame[-1]):
        return trim_recursive(frame[:-1])
    # crop left
    elif not np.sum(frame[:, 0]):
        return trim_recursive(frame[:, 1:])
    # crop right
    elif not np.sum(frame[:, -1]):
        return trim_recursive(frame[:, :-1])
    return frame

def trim_contours(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros((0,0,3))
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = frame[y:y + h, x:x + w]
    return crop

def trim_contours_exact(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.zeros((0,0,3))
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = frame[y:y + h, x:x + w]
    return crop

def trim_iterative(frame):
    for start_y in range(1, frame.shape[0]):
        if np.sum(frame[:start_y]) > 0:
            start_y -= 1
            break
    if start_y == frame.shape[0]:
        if len(frame.shape) == 2:
            return np.zeros((0,0))
        else:
            return np.zeros((0,0,0))
    for trim_bottom in range(1, frame.shape[0]):
        if np.sum(frame[-trim_bottom:]) > 0:
            break

    for start_x in range(1, frame.shape[1]):
        if np.sum(frame[:, :start_x]) > 0:
            start_x -= 1
            break
    for trim_right in range(1, frame.shape[1]):
        if np.sum(frame[:, -trim_right:]) > 0:
            break

    end_y = frame.shape[0] - trim_bottom + 1
    end_x = frame.shape[1] - trim_right + 1

    # print('iterative cropping x:{}, w:{}, y:{}, h:{}'.format(start_x, end_x - start_x, start_y, end_y - start_y))
    return frame[start_y:end_y, start_x:end_x]