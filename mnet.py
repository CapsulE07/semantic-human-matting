# -*- coding: utf-8 -*-
import tensorflow as tf
from  tensorflow.python import keras
from tensorflow.python.keras.layers import  BatchNormalization,GlobalAveragePooling2D,MaxPooling2D,UpSampling2D,Conv2D
from keras.layers.convolutional import Conv2DTranspose
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets as nets


class MNet(object):
    """ This is the TNet definition of fulfilling the SHM paper."""
    def __init__(self,  default_image_size = 400):
        self._default_image_size = default_image_size

    def preprocess(self, features):
        image_concat = features['image_concat']
        tnet_output = features['tnet_output']

        def mnet_input_clean(image_concat, tnet_output, image_size):
            image_concat = tf.image.resize_images(image_concat, [image_size, image_size])
            tnet_output =  tf.image.resize_images(tnet_output, [image_size, image_size])
            mnet_input = tf.concat([image_concat, tnet_output], 3)
            print(mnet_input.get_shape())
            return mnet_input

        mnet_input = mnet_input_clean(image_concat,tnet_output, self._default_image_size )

        features['mnet_input'] = mnet_input
        preprocessed_dict = features
        return preprocessed_dict
    
    def predict(self, preprocessed_input, is_training):
    #-------------------encoder---------------------------- 
    #--------(pretrained & trainable if selected)----------
        print("features shape: ", preprocessed_input.get_shape())
        with slim.arg_scope(nets.vgg.vgg_arg_scope()):
            net = self.vgg_16_modified(preprocessed_input)
    # #--------------decoder (trainable)----------- 
        with tf.variable_scope('decoder'):
            feature_map_size = net.get_shape()[1:3]
            print("feature_map_size: ", feature_map_size)

            x = Conv2D(512, (5, 5), activation='relu', padding='same',name='Deconv5')(net)
            
            x = UpSampling2D((2,2))(x)
            x = Conv2D(256, (5, 5), activation='relu', padding='same', name='Deconv4')(x)
            
            x = UpSampling2D((2,2))(x)
            x = Conv2D(128, (5, 5), activation='relu', padding='same', name='Deconv3')(x)
            
            x = UpSampling2D((2,2))(x)
            x = Conv2D(64, (5, 5), activation='relu', padding='same', name='Deconv2')(x)

            x = UpSampling2D((2,2))(x)
            x = Conv2D(64, (5, 5), activation='relu', padding='same', name='Deconv1')(x)
            
            x = Conv2D(1, (5, 5), activation='relu', padding='same', name='Raw_Alpha_Pred')(x)
        
        return x       
    

    def postprocess(self, preprocessed_dict, mnet_predict_output):
        tnet_output = preprocessed_dict['tnet_output']
        Fs = tnet_output[:,:,:,1]
        Us = tnet_output[:,:,:,2]
        print(Fs.get_shape())
        print(Us.get_shape())
        
        Us = tf.expand_dims(Us, axis = 3)
        print("expand_dims Us: ",Us.get_shape())
        Fs = tf.expand_dims(Fs, axis = 3)
        print("expand_dims Fs: ",Fs.get_shape())
        Us = tf.cast(Us, dtype=tf.float32)
        mnet_predict_output = tf.cast(mnet_predict_output, dtype=tf.float32)
        featured_multiplyed = tf.multiply(Us,mnet_predict_output)
        print("featured_multiplyed",featured_multiplyed.get_shape())

        Fs = tf.cast(Fs, dtype=tf.float32)
        features_fused  = tf.add(featured_multiplyed,Fs)
        print("features_fused", features_fused.get_shape())

        preprocessed_dict["features_fused"] = features_fused
        return  preprocessed_dict

    def vgg_16_modified(self, inputs,scope='vgg_16'):
        with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
            with slim.arg_scope([slim.conv2d, slim.max_pool2d]):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                return net


    def loss(self, features,labels):
        loss = tf.reduce_mean(tf.square(features - labels))
        return loss
