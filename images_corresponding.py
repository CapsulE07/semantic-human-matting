# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 21:11:27 2018

@author: shirhe-lyh
"""

import numpy as np

if __name__ == '__main__':
    images_txt_path = ('/home/jingxiong/python_project/mask_rcnn_matting/' +
                          'datasets/train_images_without_boundary_with_bg.txt')
    output_txt_path = ('/home/jingxiong/python_project/' +
                       'deep_image_matting_with_mask_rcnn/datasets_with_bg/' +
                       'images_correspondence.txt')
    trimaps_dirs = ['trimaps', 'trimaps_gt']
    
    with open(images_txt_path, 'r') as reader:
        images_names = np.loadtxt(reader, str, delimiter='@')
    
    with open(output_txt_path, 'w') as writer:
        for i, [image_fg_name, image_bg_name] in enumerate(images_names):
            trimap_name = image_fg_name.replace('/', '_')
            trimap_path = trimaps_dirs[i%2] + '/' + trimap_name
            writer.write(image_fg_name + '@' + image_bg_name + '@' +
                         trimap_path + '\n')
                        
