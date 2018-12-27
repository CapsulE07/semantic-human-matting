# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from tensorflow.python.keras.models import Sequential,Model
# -*- coding: utf-8 -*-
import tensorflow as tf



def _fixed_sides_resize(image, output_height, output_width):
    """Resize images by fixed sides.

    Args:
        image: A 3-D image `Tensor`.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.

    Returns:
        resized_image: A 3-D tensor containing the resized image.
    """
    output_height = tf.convert_to_tensor(output_height, dtype=tf.int32)
    output_width = tf.convert_to_tensor(output_width, dtype=tf.int32)

    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_nearest_neighbor(
        image, [output_height, output_width], align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image

def _tf_example_parser(serialized):
    features = {
        'image/encoded_image_fg':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded_image_bg':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded_concat_img':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/alpha_matte':
            tf.FixedLenFeature((), tf.string, default_value=''),   
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value=''), 
        'image/trimap':
            tf.FixedLenFeature((), tf.string, default_value=''),   
        'imgae/width': 
            tf.FixedLenFeature([], tf.int64),
        'imgae/height': 
            tf.FixedLenFeature([], tf.int64)}
    
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(
        serialized=serialized, features=features)

    # Get the image as raw bytes.
    image_fg = tf.image.decode_jpeg(parsed_example['image/encoded_image_fg'], channels=3)
    image_bg = tf.image.decode_jpeg(parsed_example['image/encoded_image_bg'],  channels=3)
    image_concat = tf.image.decode_jpeg(parsed_example['image/encoded_concat_img'],  channels=3)
    alpha = tf.image.decode_jpeg(parsed_example['image/alpha_matte'], channels=1)
    trimap = tf.image.decode_jpeg(parsed_example['image/trimap'],  channels=3)

#     height = tf.cast(parsed_example['imgae/height'], tf.int32)
#     width = tf.cast(parsed_example['imgae/width'], tf.int32)
        
    width,height = 400,400
    image_fg,image_bg, image_concat ,trimap, alpha = _decode_resize_normalized(image_fg,image_bg, image_concat ,trimap, alpha, height, width)
    

    return image_fg,image_bg, image_concat ,trimap, alpha


# Helper-function for creating an input-function that reads from TFRecords files for use with the Estimator API.
def input_fn(filenames, is_training=True, network='TNET', batch_size=4, buffer_size=1024):
    """input_fn 
        Create a TensorFlow Dataset-object which has functionality
        for reading and shuffling data from TFRecords files.
    Args:
        filenames:   Filenames for the TFRecords files.
        is_training: Boolean whether training (True) or testing (False).
        network: Flag which indicates which network is used
        batch_size:  Return batches of this size.
        buffer_size: Read buffers of this size. The random shuffling
                    is done on the buffer, so it must be big enough.
        
    Returns:
        a dataset object
    """

    dataset = tf.data.TFRecordDataset(filenames=filenames)

    dataset = dataset.map(_tf_example_parser)

    if network == 'TNET':
        dataset = dataset.map(_TNet_Fileter)
    elif network =='MNET':
        dataset = dataset.map(_MNet_Fileter)
    else:
        dataset = dataset.map(_Whole_Filter)
   
    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        num_repeat = 1

    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)

    return dataset



def _TNet_Fileter(image_fg,image_bg, image_concat ,trimap, alpha_matte):
    features = {"image_concat": image_concat}
    label  = trimap
    return features, label 

def _MNet_Fileter(image_fg,image_bg, image_concat ,trimap, alpha_matte):
    features = {"image_concat": image_concat, "trimap": trimap}
    label  = alpha_matte
    return features, label 

def _Whole_Filter(image_fg,image_bg, image_concat ,trimap, alpha_matte):
    features = {"image_concat": image_concat, "image_fg": image_fg, "image_bg": image_bg, "trimap": trimap}
    label  = alpha_matte
    return features, label 



def _decode_resize_normalized(image_fg,image_bg, image_concat ,trimap, alpha, height, width):

    image_fg = _image_decode_resize_normalized(image_fg,height, width, True )
    image_bg = _image_decode_resize_normalized(image_bg,height, width, True )
    image_concat = _image_decode_resize_normalized(image_concat,height, width, True )
    alpha = _image_decode_resize_normalized(alpha,height, width, False )

    trimap = _image_decode_resize_normalized(trimap,height, width, False )
    
    alpha = _alpha_reshape_cast(alpha)
    return image_fg,image_bg, image_concat ,trimap, alpha


def _alpha_reshape_cast(alpha_matte, threshhold=0.7):
    """alpha preprocessing. 

    Outputs of this function can be used as labels.

    Args: 
        alpha_matte: A float32 tensor with shape [batch_size,height, width, 1] representing a batch of groundtruth masks
    
    Returns:
        The preprocessed tensors
    """
    alpha_matte = tf.where(alpha_matte > threshhold, tf.ones_like(alpha_matte), tf.zeros_like(alpha_matte)) 
    alpha_matte = tf.cast(alpha_matte, tf.int32)
    return alpha_matte

def _image_decode_resize_normalized(image_tensor,height, weight, is_normalized=True):
    """input tensor  preprocessing. 

    Outputs of this function can be used as labels.
    Args: 
        image_tensor: A float32 tensor with shape [batch_size,height, width, 1] representing a batch of groundtruth masks
    
    Returns:
        The preprocessed tensors
    """
    image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.float32)
    image_tensor = tf.image.resize_images(image_tensor, [height, weight])
    if is_normalized:
        image_tensor /=  255
    return image_tensor


    

   
        
