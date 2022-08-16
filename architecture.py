#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:14:40 2022

@author: nanditapuri
"""




##CNN structure
# Image size 43 x 43

conv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
conv_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
max_pooling_1 = tf.keras.layers.GlobalMaxPool2D()([conv_1, conv_1])


conv_3 = tf.keras.layers.Conv2D(43, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(s)
conv_4 = tf.keras.layers.Conv2D(43, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
max_pooling_2 = tf.keras.layers.GlobalMaxPool2D()([conv_1, conv_1])


#fully connected layer