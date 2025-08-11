#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import struct

from cv_bridge import CvBridge

import numpy as np
g_node = None
cv_bridge = None




# def depth_rgb_callback(msg):
#     global g_node
#     global cv_bridge
#     g_node.get_logger().info(f'msg.data[1124044] = {msg.data[1124044]}')


#     img = cv_bridge.imgmsg_to_cv2(msg)
#     cv2.imshow("Depth->RGB Raw Image", img)
#     cv2.waitKey(1)

def unpack_rgb_float(rgb_float):
    s = struct.pack('f', rgb_float)
    i = struct.unpack('I', s)[0]
    r = (i >> 16) & 0xFF
    g = (i >> 8) & 0xFF
    b = i & 0xFF
    return r, g, b


def pointcloud_callback(msg):
    global g_node
    # g_node.get_logger().info(f'Received point cloud with {msg.width * msg.height} points.')

    # Optional: iterate through a few points (slow for large clouds)
    # gen = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
    # for i, point in enumerate(gen):
    #     x, y, z, rgb = point
    #     r, g, b = unpack_rgb_float(rgb)
    #     z = z*39.37 # convert to inches
    #     print(f'Point {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}, r={r}, g={g}, b={b}')
    #     if i > 4:
    #         break  # limit output


def main(args=None):
    global g_node
    global cv_bridge

    rclpy.init(args=args)

    g_node = rclpy.create_node('test_azure_kinect_subscriber')
    g_node.get_logger().info("Starting %s" % g_node.get_name())

    cv_bridge = CvBridge()

   
    # subscription_depth_to_rgb = g_node.create_subscription(Image, '/k4a/depth_to_rgb/image_raw', depth_rgb_callback, 10)
    # g_node.get_logger().info("Subscribed to %s" % subscription_depth_to_rgb.topic_name)

    subscription_pc = g_node.create_subscription(PointCloud2, '/k4a/points2', pointcloud_callback, 10)
    g_node.get_logger().info(f"Subscribed to {subscription_pc.topic_name}")

    while rclpy.ok():
        rclpy.spin_once(g_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    g_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
