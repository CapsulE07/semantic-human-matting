# -*- coding: utf-8 -*-
import tensorflow as tf
from  tensorflow.python import keras
from tensorflow.python.keras.layers import Conv2D,BatchNormalization,Dropout,Concatenate,Activation,AveragePooling2D
from tensorflow.python.keras import layers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets as nets




class TNet(object):
    """ This is the TNet definition of fulfilling the SHM paper."""
    def __init__(self, default_image_size = 400):
        pass
    
    def preprocess(self, features):
        preprocessed_dict = features
        return preprocessed_dict

    def predict(self, preprocessed_input,is_training):
        """TNet definition
        """
        nets  = build_pspnet50(preprocessed_input,is_training)
        return nets

    def postprocess(self, preprocessed_dict, nets):
        preprocessed_dict['tnet_output'] = nets

        '''
        post process steps
        '''
        postprocessed_dict = preprocessed_dict
        return postprocessed_dict

    def loss(self, features,label):
        pass




def build_pspnet50(features, is_training):
    input_shape = features.get_shape()[1:3]
    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = nets.resnet_v1.resnet_v1_50(features,  output_stride=4,reuse= tf.AUTO_REUSE,global_pool=False,
                                                        is_training=is_training)
    with tf.variable_scope('pyramid'):
        feature_map_size = net.get_shape()[1:3]
        print("feature_map_size: ", feature_map_size)
        psp = build_pyramid_pooling_module(net,feature_map_size)
        print(psp.get_shape())

        x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
                    use_bias=False)(psp)
        x = BN(name="conv5_4_bn")(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)
        x = Conv2D(3, (1, 1), strides=(1, 1), name="conv6")(x)
        x = Interp(input_shape)(x)
    return x



def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)
    
def build_pyramid_pooling_module(res, feature_map_size):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
#     feature_map_size = tuple(int(ceil(input_dim / 8.0))
#                              for input_dim in input_shape)
#     print("PSP module will interpolate to a final feature map size of %s" %
#           (feature_map_size, ))
    
    interp_block1 = interp_block(res, 1, feature_map_size)
    interp_block2 = interp_block(res, 2, feature_map_size)
    interp_block3 = interp_block(res, 3, feature_map_size)
    interp_block6 = interp_block(res, 6, feature_map_size)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res
    
class Interp(layers.Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = tf.image.resize_images(inputs, [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config

    
def interp_block(prev_layer, level, feature_map_shape):
    
    print("prev_layer before interp_block: ", prev_layer.get_shape())
    kernel_strides_map = {1: 60,
                          2: 30,
                          3: 20,
                          6: 10}
    names = [
        "conv5_3_pool" + str(level) + "_conv",
        "conv5_3_pool" + str(level) + "_conv_bn"
    ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0],
                        use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer) 
    prev_layer = Interp(feature_map_shape)(prev_layer)
    return prev_layer

    

        