#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import struct

class FilteredPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('filtered_pointcloud_publisher')
        self.publisher = self.create_publisher(PointCloud2, '/filtered_cloud', 10)
        self.subscription = self.create_subscription(
            PointCloud2, '/k4a/points2', self.pointcloud_callback, 10
        )
        self.get_logger().info('FilteredPointCloudPublisher node started.')

    def pointcloud_callback(self, msg):
        points = []

        for x, y, z, rgb in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            r, g, b = self.unpack_rgb_float(rgb)
            if self.is_target_color(r, g, b):
                # print(r,g,b)
                rgb_float = self.pack_rgb_float(r, g, b)
                points.append([x, y, z, rgb_float])

        if not points:
            # self.get_logger().warn("No points matched the color filter. Skipping publish.")
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'depth_camera_link'  # Should match a known TF frame

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        pc_msg = pc2.create_cloud(header, fields, points)
        self.publisher.publish(pc_msg)

    def unpack_rgb_float(self, rgb_float):
        s = struct.pack('f', rgb_float)
        i = struct.unpack('I', s)[0]
        r = (i >> 16) & 0xFF
        g = (i >> 8) & 0xFF
        b = i & 0xFF
        return r, g, b

    def pack_rgb_float(self, r, g, b):
        i = (r << 16) | (g << 8) | b
        return struct.unpack('f', struct.pack('I', i))[0]

    def is_target_color(self, r, g, b):
        # Example: keep bright blue points
        return r < 11 and g > 120 and 200 > b > 30 # r < 11 and g > 120 and 200 > b > 30
 

def main(args=None):
    rclpy.init(args=args)
    node = FilteredPointCloudPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
