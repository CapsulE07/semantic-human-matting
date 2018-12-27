# -*- coding: utf-8 -*-
import tensorflow as tf

def _tf_example_parser(serialized):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = {
        'image_fg/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'trimap/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'alpha_matte/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''), 
        'image_bg/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),   
        'shape': 
            tf.FixedLenFeature(shape=(3,), dtype=tf.int64)}   

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(
        serialized=serialized, features=features)

    image_fg = tf.image.decode_jpeg(parsed_example['image_fg/encoded'])
    trimap = tf.image.decode_jpeg(parsed_example['image_fg/encoded'])
    alpha_matte = tf.image.decode_jpeg(parsed_example['image_fg/encoded'])
    image_bg = tf.image.decode_jpeg(parsed_example['image_bg/encoded'])
    print("----------------------------------------")
    print(tf.shape(image_fg))
    print(image_fg.get_shape)
    # shape = parsed_example['shape']
    # image_fg = tf.reshape(image_fg,shape=shape)

    image_fg = _fixed_sides_resize(image_fg, 320,320)
    image_bg = _fixed_sides_resize(image_bg, 320,320)
    return image_fg, image_bg, alpha_matte, trimap







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


def input_fn(data_path, batch_size, num_steps, is_training=True):
    """
    Parse image and data in tfrecords file for training
    Args:
        data_path:  a list or single tf records path
        batch_size: size of images returned
        is_training: is training stage or not
    Returns:
        image and labels batches of randomly shuffling tensors
    """
    if not isinstance(data_path, (tuple, list)):
        data_path = [data_path]
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_tf_example_parser)
    dataset = dataset.batch(batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=50000)
        dataset = dataset.repeat(num_steps)

    iterator = dataset.make_one_shot_iterator()

    image_fg, image_bg, alpha_matte, trimap = iterator.get_next()

    # # convert to onehot label
    # labels = tf.one_hot(labels, 32)  # 二分类
    # # preprocess image: scale pixel values from 0-255 to 0-1
    # images = tf.image.convert_image_dtype(images, dtype=tf.float32)  # 将图片像素从0-255转换成0-1，tf提供的图像操作大多需要0-1之间的float32类型
    # images /= 255.
    # images -= 0.5
    # images *= 2.
    # print("tf.shape:" , tf.shape(images))
    # print("...get_shape", images.get_shape())
    return image_fg, image_bg, alpha_matte, trimap





# def get_record_dataset(record_path,
#                        reader=None, 
#                        num_samples=50000, 
#                        num_classes=7):
#     """Get a tensorflow record file.
    
#     Args:
        
#     """
#     if not reader:
#         reader = tf.TFRecordReader
        
#     keys_to_features = {
#         'image_fg/encoded':
#             tf.FixedLenFeature((), tf.string, default_value=''),
#         'image/format':
#             tf.FixedLenFeature((), tf.string, default_value='jpeg'),
#         'trimap/encoded':
#             tf.FixedLenFeature((), tf.string, default_value=''),
#         'alpha_matte/encoded':
#             tf.FixedLenFeature((), tf.string, default_value=''), 
#         'image_bg/encoded':
#             tf.FixedLenFeature((), tf.string, default_value=''),   
#         'shape': 
#             tf.FixedLenFeature(shape=(3,), dtype=tf.int64)}   
        
#     items_to_handlers = {
#         'image': slim.tfexample_decoder.Image(image_key='image/encoded',
#                                               format_key='image/format'),
#         'image': slim.tfexample_decoder.Image(image_key='image/encoded',
#                                               format_key='image/format'),
#         'image': slim.tfexample_decoder.Image(image_key='image/encoded',
#                                               format_key='image/format'),
#         'label': slim.tfexample_decoder.Tensor('shape', shape=[3,])}
#     decoder = slim.tfexample_decoder.TFExampleDecoder(
#         keys_to_features, items_to_handlers)
    
#     labels_to_names = None
#     items_to_descriptions = {
#         'image': 'An image with shape image_shape.',
#         'label': 'A single integer.'}
#     return slim.dataset.Dataset(
#         data_sources=record_path,
#         reader=reader,
#         decoder=decoder,
#         num_samples=num_samples,
#         num_classes=num_classes,
#         items_to_descriptions=items_to_descriptions,
#         labels_to_names=labels_to_names)






if __name__ == "__main__":
    record_path  = "/data2/dongkuiyao/matting/tfrecord/train_boundary.record"
    image_fg, image_bg, alpha_matte, trimap = input_fn(record_path, 16, 20000)
    with tf.Session() as sess:
        image_fg_sess, image_bg_sess, alpha_matte_sess, trimap_sess = sess.run([image_fg, image_bg, alpha_matte, trimap])
        print(image_fg_sess.shape)