#!/usr/bin/env python3
import sys
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import tf_transformations

# Add DROID-SLAM to path
DROID_PATH = "/home/garrett/VSCode/EECE7150-final2/DROID-SLAM"
sys.path.append(DROID_PATH)
sys.path.append(os.path.join(DROID_PATH, "droid_slam"))

from droid import Droid

class DroidArgs:
    def __init__(self, weights_path):
        self.weights = weights_path
        self.buffer = 1024
        self.image_size = [376, 672] # Will be updated on first frame
        self.disable_vis = True
        self.stereo = False
        self.beta = 0.3
        self.filter_thresh = 2.4
        self.warmup = 8
        self.keyframe_thresh = 4.0
        self.frontend_thresh = 16.0
        self.frontend_window = 25
        self.frontend_radius = 2
        self.frontend_nms = 1
        self.backend_thresh = 22.0
        self.backend_radius = 2
        self.backend_nms = 3
        self.upsample = True
        self.asynchronous = False # Run synchronously for simplicity

class DroidSlamNode(Node):
    def __init__(self):
        super().__init__('droid_slam_node')
        
        self.declare_parameter('weights', os.path.join(DROID_PATH, 'droid.pth'))
        self.declare_parameter('image_size', [376, 672])
        
        weights_path = self.get_parameter('weights').value
        self.args = DroidArgs(weights_path)
        
        self.bridge = CvBridge()
        self.droid = None
        self.t = 0
        
        # Subscribers
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth_registered/image_raw', 10)
        
        self.latest_camera_info = None
        self.get_logger().info("DROID-SLAM Node Initialized")

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def image_callback(self, msg):
        if self.latest_camera_info is None:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # Initialize DROID on first frame
        if self.droid is None:
            h, w = cv_image.shape[:2]
            self.args.image_size = [h, w]
            self.droid = Droid(self.args)
            self.get_logger().info(f"DROID-SLAM initialized with image size: {h}x{w}")

        # Prepare input for DROID
        # Image: [1, 3, H, W]
        image_tensor = torch.as_tensor(cv_image).permute(2, 0, 1).float()
        image_tensor = image_tensor[None].cuda()
        
        # Intrinsics: [1, 4] -> [fx, fy, cx, cy]
        K = self.latest_camera_info.k
        intrinsics = torch.as_tensor([K[0], K[4], K[2], K[5]]).float()
        intrinsics = intrinsics[None].cuda()
        
        # Track
        self.droid.track(self.t, image_tensor, intrinsics=intrinsics)
        
        # Extract Pose and Depth
        # DROID stores state in self.droid.video
        # Current index is self.droid.video.counter.value - 1 (since track increments it? No, track uses t)
        # Actually track calls filterx.track which appends to video.
        
        # Let's get the latest pose from video
        idx = self.droid.video.counter.value - 1
        if idx < 0:
            return

        # Pose: [x, y, z, qx, qy, qz, qw] (or similar, need to verify order)
        # DROID uses [tx, ty, tz, qx, qy, qz, qw]
        pose = self.droid.video.poses[idx].cpu().numpy()
        
        # Publish Odometry
        odom_msg = Odometry()
        odom_msg.header = msg.header
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link" # DROID tracks camera, which we assume is base_link
        
        odom_msg.pose.pose.position.x = float(pose[0])
        odom_msg.pose.pose.position.y = float(pose[1])
        odom_msg.pose.pose.position.z = float(pose[2])
        odom_msg.pose.pose.orientation.x = float(pose[3])
        odom_msg.pose.pose.orientation.y = float(pose[4])
        odom_msg.pose.pose.orientation.z = float(pose[5])
        odom_msg.pose.pose.orientation.w = float(pose[6])
        
        self.odom_pub.publish(odom_msg)
        
        # Extract and Publish Depth
        # Disparity is in self.droid.video.disps_up[idx]
        # Depth = 1 / Disparity
        disp = self.droid.video.disps_up[idx].cpu().numpy()
        depth = 1.0 / np.maximum(disp, 0.001)
        
        depth_msg = self.bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding="32FC1")
        depth_msg.header = msg.header
        self.depth_pub.publish(depth_msg)
        
        self.t += 1

def main(args=None):
    rclpy.init(args=args)
    node = DroidSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Terminate DROID to save reconstruction if needed (optional)
        # node.droid.terminate(...)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
