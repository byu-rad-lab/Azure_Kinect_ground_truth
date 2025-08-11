#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import struct
import numpy as np
# from UnivariateSpline import Curve_fit2
import time
import os




class CollectInfo(Node):

    def __init__(self):
        super().__init__('PC_extractor')

        self.subscription = self.create_subscription(
            PointCloud2, '/filtered_cloud', self.pointcloud_callback, 10
        )
        self.num = 0
        self.get_logger().info('PC_extractor node started')

    def pointcloud_callback(self, msg):
        self.get_logger().info('Collecting points')
        points = list(pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True)) # list(np.array([(x1,y1,z1,color1), (x2,y2,z2,color2), ...)]))
        xyz = np.array(points)
        path = os.path.expanduser('~/ros2_ws/src/azure_listener/azure_listener/filtered_data_active.txt')
        np.savetxt(path, xyz, delimiter = ',')
        # self.num += 1

        







def main(args=None):
    rclpy.init(args=args)
    node = CollectInfo()
    
    i = 0
    while True:
        node.get_logger().info(f'Analyzing the information, {i}')
        rclpy.spin_once(node)
        time.sleep(1)
        i += 1

    node.get_logger().info('Destroying node')
    node.destroy_node()
    rclpy.shutdown()




if __name__ == '__main__':
    main()