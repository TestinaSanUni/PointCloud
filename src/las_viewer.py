import laspy
import open3d as o3d
import numpy as np
import os
import sys

if len(sys.argv) < 2:
    print("\nERRORE: Nessun file LAS specificato.")
    sys.exit(1)

las_file_path = sys.argv[1]

las = laspy.read(las_file_path)

points = np.vstack((las.x, las.y, las.z)).T

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

if hasattr(las, "red"):
    colors = np.vstack((las.red, las.green, las.blue)).T
    colors = colors / 65535.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])