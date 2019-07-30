#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:51:34 2019

@author: Narahari B M
"""

import numpy as np
import cv2


def get_custom_augmentations(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255,
                               random_crop_size=(32, 32), padding_for_crop=4,pixel_level=False,
                               mean_norm = [0.4914, 0.4822, 0.4465],
                               std_norm = [0.2023, 0.1994, 0.2010],
                               normalize = True,
                               random_crop = False,
                               cutout = False
                               ):
    
    """
    This function wraps all the custom data augmentation functions
    Current functionalities:
        normalize channel wise
        random crop with padding added
        cutout
    """
    
    def normalize(inp_img):
        inp_img[:,:, 0] -= mean_norm[0]
        inp_img[:,:, 1] -= mean_norm[1]
        inp_img[:,:, 1] -= mean_norm[1]
        
        inp_img[:,:, 0] /= std_norm[0]
        inp_img[:,:, 1] /= std_norm[1]
        inp_img[:,:, 1] /= std_norm[1]
        
        return inp_img
        
  
    def eraser(input_img):
        """
        Function for random cutout
        """
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img
      
    def random_crop(input_img):
        
        """
        Function for random crop with padding
        """
        
        # Note: image_data_format is 'channel_last'
        assert input_img.shape[2] == 3
        
        img = cv2.copyMakeBorder(input_img, padding_for_crop, padding_for_crop, padding_for_crop, padding_for_crop, cv2.BORDER_REPLICATE)
        
        height, width = img.shape[0], img.shape[1]
        
        dy, dx = random_crop_size
        
        x = np.random.randint(0, width - dx + 1)
        
        y = np.random.randint(0, height - dy + 1)
        
        return img[y:(y+dy), x:(x+dx), :]
      
    def do_preproc(input_image):
        
        """
        Combine all data augmentation process
        """
        proc_image = input_image
        if normalize:
            proc_image = normalize(proc_image)
        if random_crop:
            proc_image = random_crop(proc_image)
        if cutout:
            proc_image = eraser(proc_image)
      
        return proc_image
      
    return do_preproc