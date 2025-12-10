#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class DepthStats(Node):
    def __init__(self):
        super().__init__('depth_stats')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/camera/depth_registered/image_raw', self.callback, 10)
        self.get_logger().info("Listening for depth images...")

    def callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Filter valid depths (non-zero, non-inf)
            valid = depth[np.isfinite(depth)]
            valid = valid[valid > 0]
            
            if len(valid) == 0:
                self.get_logger().warn("Depth image is empty or all zeros/NaNs!")
                return

            min_d = np.min(valid)
            max_d = np.max(valid)
            mean_d = np.mean(valid)
            
            self.get_logger().info(f"Depth Stats (m): Min={min_d:.2f}, Max={max_d:.2f}, Mean={mean_d:.2f}")
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthStats()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
