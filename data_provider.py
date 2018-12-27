# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:44:07 2018

@author: shirhe-lyh
"""

import cv2
import numpy as np
import os


def alpha_matte(image):
    """Returns the alpha channel of a given image."""
    if image.shape[2] > 3:
        alpha = image[:, :, 3]
        alpha = np.where(alpha > 0, 1, 0)
    else:
        reduced_image = np.sum(np.abs(255 - image), axis=2)
        alpha = np.where(reduced_image > 100, 1, 0)
    alpha = alpha.astype(np.uint8)
    return alpha


def trimap(mask, mode='mask', boundary_width=50,
           kernel_size_low=50, kernel_size_high=150):
    """Returns the trimap of a given mask."""
    if mode == 'trivial':
        return np.ones_like(mask, dtype=np.uint8) * 128
    elif mode == 'boundary':
        trimap_b = np.ones_like(mask, dtype=np.uint8) * 128
        trimap_b[:boundary_width] = 0
        trimap_b[-boundary_width:] = 0
        trimap_b[:, :boundary_width] = 0
        trimap_b[:, -boundary_width:] = 0
        return trimap_b

    erode_kernel_size = np.random.randint(kernel_size_low, kernel_size_high)
    dilate_kernel_size = np.random.randint(kernel_size_low, kernel_size_high)
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (erode_kernel_size, erode_kernel_size))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
    eroded_alpha = cv2.erode(mask, erode_kernel)
    dilated_alpha = cv2.dilate(mask, dilate_kernel)

    trimap_d = np.where(dilated_alpha > 0, 128, 0)
    trimap_e = np.where(eroded_alpha > 0, 127, 0)
    trimap_sum = trimap_d + trimap_e
    trimap_sum = trimap_sum.astype(np.uint8)
    return trimap_sum


def image(image):
    """Returns the 3-channel composite image."""
    image_fg = image[:, :, :3]
    alpha = alpha_matte(image)
    alpha_expanded = np.expand_dims(alpha, axis=2)
    b, g, r = np.split(image_fg, 3, axis=2)
    b = b * alpha_expanded + (1 - alpha_expanded) * 255
    g = g * alpha_expanded + (1 - alpha_expanded) * 255
    r = r * alpha_expanded + (1 - alpha_expanded) * 255
    image_composited = np.concatenate([r, g, b], axis=2)
    return image_composited


def provide(txt_path, images_fg_dir=None, images_bg_dir=None):
    """Returns the paths of images.

    Args:
        txt_path: A .txt file with format:
            [image_fg_0, image_bg_0,
             image_fg_1, image_bg_1,
             ...]
            representing the 1-1 correspondence of foreground and background
            images.
        images_fg_dir: Path to the foreground images directory.
        images_bg_dir: Path to the background images directory.
        trimaps_dir: Path to the trimaps directory.

    Returns:
        The paths of foreground and background images.
    """
    if not os.path.exists(txt_path):
        raise ValueError('`txt_path` does not exist.')

    with open(txt_path, 'r') as reader:
        txt_content = np.loadtxt(reader, str, delimiter='@')
        np.random.shuffle(txt_content)
    if images_fg_dir is None and images_bg_dir is None:
        return txt_content
    image_paths = []
    for image_fg_rel_path, image_bg_rel_path in txt_content:
        image_fg_abs_path = image_fg_rel_path
        image_bg_abs_path = image_bg_rel_path
        if images_fg_dir is not None:
            image_fg_abs_path = os.path.join(images_fg_dir, image_fg_rel_path)
        if images_bg_dir is not None:
            image_bg_abs_path = os.path.join(images_bg_dir, image_bg_rel_path)

        image_paths.append([image_fg_abs_path, image_bg_abs_path])
    return image_paths
