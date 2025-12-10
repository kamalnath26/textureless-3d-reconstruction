#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthAnythingNode(Node):
    def __init__(self):
        super().__init__('depth_anything_node')
        
        # params
        self.declare_parameter('model_id', 'LiheYoung/depth-anything-base-hf')
        self.declare_parameter('compute_device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('max_depth', 3.5) # max depth in meters, 0 to disable
        self.declare_parameter('depth_scale_factor', 20.0) # scale factor for inverse depth
        
        model_id = self.get_parameter('model_id').value
        self.device = self.get_parameter('compute_device').value
        
        self.get_logger().info(f"Loading Depth Anything model: {model_id} on {self.device}...")
        
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
            self.get_logger().info("Model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise e

        self.bridge = CvBridge()
        
        # subs
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        
        # pubs
        self.depth_pub = self.create_publisher(Image, '/camera/depth_registered/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/depth_registered/camera_info', 10)
        
        self.latest_camera_info = None

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def image_callback(self, msg):
        if self.latest_camera_info is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # inference
        inputs = self.image_processor(images=cv_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=cv_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )

        # normalize depth for visualization/usage
        depth = prediction.squeeze().cpu().numpy()
        depth = np.maximum(depth, 0.001)
        
        scale_factor = self.get_parameter('depth_scale_factor').value
        metric_depth = scale_factor / depth 
        
        # filter far-away points to mimic a real depth camera
        # set pixels with depth > max_depth to 0 (invalid/infinity)
        max_depth = self.get_parameter('max_depth').value
        if max_depth > 0:
            metric_depth[metric_depth > max_depth] = 0.0
        
        # convert to 32FC1 (meters)
        depth_image = metric_depth.astype(np.float32)

        # publish depth
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
        depth_msg.header = msg.header
        self.depth_pub.publish(depth_msg)
        
        # publish camera info (synced)
        self.latest_camera_info.header = msg.header
        self.camera_info_pub.publish(self.latest_camera_info)

def main(args=None):
    rclpy.init(args=args)
    node = DepthAnythingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
