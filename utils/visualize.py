# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# A stand-alone visualization tool for point cloud with Open3D

import numpy as np
import open3d as o3d
import argparse
import os

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif val.lower() in ('false', 'no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expect a Boolean value.')


def read_point_cloud(file_name):
    """Read a point cloud from the specified file, support both "ply" and "bin"."""

    pcd = None
    if os.path.splitext(file_name)[1].lower() == '.ply':
        pcd = o3d.io.read_point_cloud(file_name)
    elif os.path.splitext(file_name)[1].lower() == '.bin':
        xyz = np.fromfile(file_name, dtype=np.float32).reshape((-1, 4))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[:,:3])

    return pcd


def sphere_pc_generator(opt):
    """Generate the point cloud geometry with sphere decorator."""

    pcd = read_point_cloud(opt.file_name)
    points = np.asarray(pcd.points)
    spheres = []
    
    # Create the balls one-by-one 
    for cnt, xyz in enumerate(points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(opt.radius, resolution=20) # create a ball
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(opt.color) # paint the ball
        sphere.translate(xyz, False) # translate it
        spheres.append(sphere)

    return spheres


def main():

    # Handle the radius
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=opt.window_name, height=opt.window_height, width=opt.window_width)

    if opt.radius > 0: # Render the point cloud with ball decorator
        pc_elem = sphere_pc_generator(opt)

        # Aggregate the generated spheres
        geo = pc_elem[0]
        for i in range(1, len(pc_elem)):
            geo += pc_elem[i]
        print('Aggregated %d shperes for the point cloud.' % len(pc_elem))
    else:
        geo = read_point_cloud(opt.file_name)
        print('Loaded a point cloud with %d points.' % len(geo.points))

    # Draw the stuff finally
    vis.add_geometry(geo) # Add the point cloud

    # Mark the origin if needed
    if opt.radius_origin > 0:
        origin = o3d.geometry.TriangleMesh.create_sphere(opt.radius_origin, resolution=20) # create a ball
        origin.compute_vertex_normals()
        origin.paint_uniform_color(opt.color) # paint the ball
        origin.translate([0, 0, 0], False) # translate it
        vis.add_geometry(origin) # Add the origin

    ctr = vis.get_view_control()
    if opt.view_file != '.': # Set the camera view point
        param = o3d.io.read_pinhole_camera_parameters(opt.view_file)
        ctr.convert_from_pinhole_camera_parameters(param)

    # Render and save as an image if the ouput file path is given
    if opt.output_file != '.':
        vis.capture_screen_image(opt.output_file, True)
    else:
        vis.run()
    vis.destroy_window()


def add_options(parser):
    parser.add_argument('--file_name', type=str, required=True, help='File name of the point cloud.')
    parser.add_argument('--output_file', type=str, default='.', help='Output file name for the rendered image.')
    parser.add_argument('--view_file', type=str, default='.', help='View point file for rendering.')
    parser.add_argument('--radius', type=float, default=-1, help='Radius of the rendered points. If > 0, render each point as a ball.')
    parser.add_argument('--color', type=float, nargs='+', default=[0.2, 0.2, 0.2], help='Specify the color of the rendered point cloud if ball decorator is used.')
    parser.add_argument('--radius_origin', type=float, default=-1, help='Radius of the origin points. If < 0, do not add origin.')
    parser.add_argument('--window_name', type=str, default='Point Cloud', help='Window name.')
    parser.add_argument('--window_height', type=int, default=1200, help='Window height.')
    parser.add_argument('--window_width', type=int, default=1600, help='Window width.')

    return parser


if __name__ == "__main__":

    # Initialize parser with basic options
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_options(parser)
    opt, _ = parser.parse_known_args()
    main()