#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:21:57 2022

@author: nanditapuri
"""

### DCGM + MASK RCNN ------
## An automatic nucelei segmentation method based on deep convolution neural network for histopathology images



# DCGMM -- Deep convolution Gaussian mixture color normalization mode ---
##-- to reduce color variations in hitopathology images

**** Need to check the role of S
** Need to double check all the conv numberings



### ------ Architecture Code ------ ####
#Expansion Path   -1st part

conv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
conv_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_1)
pool1 = tf.keras.layers.MaxPooling2D((2, 2))([conv_1, conv_2])


conv2_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
conv2_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
pool2 = tf.keras.layers.MaxPooling2D((2, 2))([conv2_1, conv2_2])

conv3_1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
conv3_2 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
conv3_3 = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
pool3 = tf.keras.layers.MaxPooling2D((2, 2))([conv3_1, conv3_2,conv3_3])

conv4_1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
conv4_2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
conv4_3 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
pool4 = tf.keras.layers.MaxPooling2D((2, 2))([conv4_1, conv4_2,conv4_3])

conv5_1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
conv5_2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
conv5_3 = tf.keras.layers.Conv2D(256, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
pool5 = tf.keras.layers.MaxPooling2D((2, 2))([conv5_1, conv5_2,conv5_3])



## Contraction Path --- 2

upsample1 = tf.keras.layers.Conv2DTranspose(256, (1, 1), strides=(2, 2), padding='same')(conv5_3)  ## Need to check if we need stride as it indicates max pooling which is not mentioned in the table
concat1 = tf.keras.layers.concatenate([upsample1, conv4_3])
conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')([upsample1, concat1])

upsample2 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv6)
concat2 = tf.keras.layers.concatenate([upsample2, conv3_3])
conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')([upsample2, concat2])

upsample3 = upsample2 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
concat3 = tf.keras.layers.concatenate([upsample3, conv2_2])
conv6 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')([upsample3, concat3])

upsample4 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
concat4 = tf.keras.layers.concatenate([upsample4, conv1_2])
conv9_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')([upsample4, concat4])
conv9_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')([conv9_1])
conv9_3 = tf.keras.layers.Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same')([conv9_2])




####---Mask RC-NN - Model 2 for segmentation


"             
https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/inspect_nucleus_model.ipynb
"







