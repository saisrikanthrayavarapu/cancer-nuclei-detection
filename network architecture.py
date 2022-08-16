#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:01:36 2022

@author: nanditapuri
"""

#### DCAN
#### Deep contour aware networks for object instance segmentation from histology images

##Downsampling

##Block1

block_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(IMG_input)
max_pooling1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block_1)


##Block2

block_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv1')(max_pooling1)
max_pooling2 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block_2)

##block3

block_3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv1')(max_pooling2)
max_pooling3 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block_3)


##block4

block_4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block1_conv1')(max_pooling3)
max_pooling4 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block_4)



##block5

block_5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block1_conv1')(max_pooling4)
max_pooling5 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block_5)

##block6

block_5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block1_conv1')(max_pooling5)




##Upsampling

upsample3 = tf.keras.layers.Conv2DTranspose(256, (1, 1), strides=(2, 2), padding='same')(block_4) 

upsample2 = tf.keras.layers.Conv2DTranspose(256, (1, 1), strides=(2, 2), padding='same')(block_5) 

upsample1 = tf.keras.layers.Conv2DTranspose(256, (1, 1), strides=(2, 2), padding='same')(block_6) 


## Not getting the stride and image size for convultion part












