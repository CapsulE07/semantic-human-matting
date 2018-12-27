#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import data_provider
import tensorflow as tf
import PIL.Image
import numpy as np
import io
import cv2
"""
@author: Dominic
"""

"""Generate tfrecord file from images.

Example Usage:
---------------
python3 train.py \
    --images_dir: Path to images (directory).
    --annotation_path: Path to annotatio's .txt file.
    --output_path: Path to .record.
    --resize_side_size: Resize images to fixed size.
"""


flags = tf.app.flags

flags.DEFINE_string('images_fg_dir',
                    '/data2/raycloud/matting_resize',
                    'Path to images (directory).')
flags.DEFINE_string('images_bg_dir',
                    '/data2/raycloud/matting_bg',
                    'Path to images (directory).')
flags.DEFINE_string('annotation_path',
                    '/home/dongkuiyao/dl_project/matting/fg_bg_mapping.txt',
                    'Path to fg_bg_mapping`s .txt file.')
flags.DEFINE_string('output_path',
                    '/data2/dongkuiyao/matting/tfrecord/train.record',
                    'Path to output tfrecord file.')
flags.DEFINE_integer('resize_side_size', 512, 'Resize images to fixed size.')

FLAGS = flags.FLAGS


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(image_fg_path, image_bg_path, resize_size=None):
    image_fg = cv2.imread(image_fg_path, -1)
    if image_fg is None:
        print(image_fg_path)
        return None
    #image_fg = cv2.imdecode(np.fromfile(image_fg_path, dtype=np.uint8), -1)
    image_bg = cv2.imread(image_bg_path)

    # Resize
    width, height, fg_shape = image_fg.shape
    if fg_shape != 4:
        return None
    # Resize
    if resize_size is not None:
        if width > height:
            width = int(width * resize_size / height)
            height = resize_size
        else:
            width = resize_size
            height = int(height * resize_size / width)
        image_fg = cv2.resize(image_fg, (width, height))
        image_bg = cv2.resize(image_bg, (width, height))

    else:
        image_bg = cv2.resize(image_bg, (width, height))


    alpha_matte = image_fg[:,:,3]

    temp = get_alpha_matte(image_fg)

    temp_alpha_matte = np.dstack((temp,temp, temp))
    concat_img = temp_alpha_matte * image_fg[:,:,:3] + (1 - temp_alpha_matte) * image_bg

    pil_concat_img = PIL.Image.fromarray(concat_img)
    bytes_io = io.BytesIO()
    pil_concat_img.save(bytes_io, format='JPEG')
    encoded_concat_img = bytes_io.getvalue()


    pil_image_fg = PIL.Image.fromarray(image_fg[:,:,:3] )
    bytes_io = io.BytesIO()
    pil_image_fg.save(bytes_io, format='JPEG')
    encoded_image_fg = bytes_io.getvalue()

    pil_image_bg = PIL.Image.fromarray(image_bg )
    bytes_io = io.BytesIO()
    pil_image_bg.save(bytes_io, format='JPEG')
    encoded_image_bg = bytes_io.getvalue()

    pil_alpha_matte= PIL.Image.fromarray(alpha_matte)
    bytes_io = io.BytesIO()
    pil_alpha_matte.save(bytes_io, format='JPEG')
    encoded_alpha_matte = bytes_io.getvalue()


    trimap = get_trimap(alpha_matte)
    
    pil_trimap= PIL.Image.fromarray(trimap)
    bytes_io = io.BytesIO()
    pil_trimap.save(bytes_io, format='JPEG')
    encoded_trimap = bytes_io.getvalue()

    tf_example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/encoded_image_fg': bytes_feature(encoded_image_fg),
            'image/encoded_image_bg': bytes_feature(encoded_image_bg),
            'image/encoded_concat_img': bytes_feature(encoded_concat_img),
            'image/alpha_matte': bytes_feature(encoded_alpha_matte), 
            'image/trimap': bytes_feature(encoded_trimap),          
            'image/format': bytes_feature('jpg'.encode()),
            'imgae/width': int64_feature(width),
            'imgae/height': int64_feature(height)}))
    return tf_example


def get_trimap(alpha_matte):
    kernel = np.ones((3,3),np.uint8) 
    erosion = cv2.erode(alpha_matte,kernel,iterations = 1)
    dilation = cv2.dilate(alpha_matte,kernel,iterations = 1)
    trimap = (erosion + dilation) /2
    single_trimap_f = np.where(trimap > 220, 1, 0)
    single_trimap_b = np.where(trimap < 50, 1, 0)
    single_trimap_u = np.where(((80 < trimap) & (trimap < 150)), 1, 0)
    single_trimap_Tnet = np.dstack([single_trimap_f, single_trimap_b, single_trimap_u])
    # print(trimap_Tnet.shape)
    # trimap_Tnet = np.dstack([trimap_Tnet,single_trimap_Tnet ])
    # print(trimap_Tnet.shae)
    return single_trimap_Tnet


def get_alpha_matte(image):
    """Returns the alpha channel of a given image."""
    if image.shape[2] > 3:
        alpha = image[:, :, 3]
        alpha = np.where(alpha > 0, 1, 0)
    else:
        reduced_image = np.sum(np.abs(255 - image), axis=2)
        alpha = np.where(reduced_image > 100, 1, 0)
    alpha = alpha.astype(np.uint8)
    return alpha


def generate_tfrecord(image_paths, output_path, resize_size=None):
    num_valid_tf_example = 0
    writer = tf.python_io.TFRecordWriter(output_path)
    for image_fg_path, image_bg_path in image_paths:
        if not tf.gfile.GFile(image_fg_path):
            print('%s does not exist.' % image_fg_path)
            continue
        if not tf.gfile.GFile(image_bg_path):
            print('%s does not exist.' % image_bg_path)
            continue
        tf_example = create_tf_example(image_fg_path, image_bg_path,
                                       resize_size)
        if tf_example is None:
            continue
        writer.write(tf_example.SerializeToString())
        num_valid_tf_example += 1

        if num_valid_tf_example % 100 == 0:
            print('Create %d TF_Example.' % num_valid_tf_example)
    writer.close()
    print('Total create TF_Example: %d' % num_valid_tf_example)
    print('The number of skiped images: %d' % (len(image_paths) -
                                               num_valid_tf_example))


def main(_):
    images_fg_dir = FLAGS.images_fg_dir
    images_bg_dir = FLAGS.images_bg_dir
    annotation_path = FLAGS.annotation_path
    record_path = FLAGS.output_path
    resize_size = FLAGS.resize_side_size

    image_paths = data_provider.provide(annotation_path, images_fg_dir,
                                        images_bg_dir)

    generate_tfrecord(image_paths, record_path, resize_size)


if __name__ == '__main__':
    tf.app.run()
