# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Convert a LiDAR point cloud to a range image based on spherical coordinate conversion

import numpy as np


def cart2spherical(input_xyz):
    """Conversion from Cartisian coordinates to spherical coordinates."""

    r = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2 + input_xyz[:, 2] ** 2)
    alpha = np.arctan2(input_xyz[:, 1], input_xyz[:, 0]) # corresponding to width
    epsilon = np.arcsin(input_xyz[:, 2] / r) # corrsponding to height
    return np.stack((r, alpha, epsilon), axis = 1)


def spherical2cart(input_spherical):
    """Conversion from spherical coordinates to Cartesian coordinates."""

    x = input_spherical[:, 0] * np.cos(input_spherical[:, 1]) * np.cos(input_spherical[:, 2])
    y = input_spherical[:, 0] * np.sin(input_spherical[:, 1]) * np.cos(input_spherical[:, 2])
    z = input_spherical[:, 0] * np.sin(input_spherical[:, 2])
    return np.stack((x, y, z), axis=1)


def pc2img(h_fov, v_fov, width, height, inf, data):
    """Convert a point cloud to an 2D image."""

    data_spherical = cart2spherical(data)

    # Project the point cloud onto an image!
    x = (data_spherical[:, 1] - h_fov[0]) / (h_fov[1] - h_fov[0])
    y = (data_spherical[:, 2] - v_fov[0]) / (v_fov[1] - v_fov[0])
    x = np.round(x * (width - 1)).astype(np.int32)
    y = np.round(y * (height - 1)).astype(np.int32)

    # exclude the pixels that are out of the selected FOV
    mask = ~((x < 0) | (x >= width) | (y < 0) | (y >= height))
    x, y = x[mask], y[mask]
    range = data_spherical[:, 0][mask]
    data_img = np.ones((height, width), dtype = np.float32) * inf
    data_img[y, x] = range

    return data_img


def img2pc(h_fov, v_fov, width, height, inf, data):
    """Convert an 2D image back to the point cloud."""
    
    alpha = (np.arange(width) / (width - 1)) * (h_fov[1] - h_fov[0]) + h_fov[0]
    epsilon = (np.arange(height) / (height - 1)) * (v_fov[1] - v_fov[0]) + v_fov[0]
    alpha, epsilon = np.meshgrid(alpha, epsilon)
    data_pc = np.stack((data, alpha, epsilon), axis=2)
    data_pc = data_pc.reshape(-1, 3)
    data_pc = data_pc[data_pc[:, 0] < inf - 1, :]
    data_pc = spherical2cart(data_pc)

    return data_pc