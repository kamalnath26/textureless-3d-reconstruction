#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2

class SimpleCameraNode(Node):
    def __init__(self):
        super().__init__('simple_camera_node')
        
        self.declare_parameter('video_device', 0)
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('frame_id', 'camera')
        self.declare_parameter('framerate', 30.0)
        self.declare_parameter('video_path', '') # Path to video file (optional)
        
        self.device_id = self.get_parameter('video_device').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.frame_id = self.get_parameter('frame_id').value
        self.framerate = self.get_parameter('framerate').value
        self.video_path = self.get_parameter('video_path').value
        
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        self.cap = None
        self.open_camera()
        
        self.timer = self.create_timer(1.0/self.framerate, self.timer_callback)

    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
            
        if self.video_path:
            self.get_logger().info(f"Opening video file: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_idx = 0
            self.direction = 1
        else:
            self.get_logger().info(f"Opening camera device {self.device_id}...")
            self.cap = cv2.VideoCapture(self.device_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open video source!")
            return

        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.get_logger().info(f"Source opened: {actual_width}x{actual_height} @ {actual_fps} FPS")

    def timer_callback(self):
        if self.cap is None or not self.cap.isOpened():
            self.open_camera()
            return

        if self.video_path:
            # Ping-pong loop logic
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                self.frame_idx += self.direction
                if self.frame_idx >= self.total_frames - 1:
                    self.direction = -1
                elif self.frame_idx <= 0:
                    self.direction = 1
            else:
                # If read failed (e.g. end of file mismatch), reverse
                self.direction *= -1
                self.frame_idx += self.direction
        else:
            # Webcam logic
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Failed to capture frame, attempting to reconnect...")
                self.open_camera()
                return

        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            self.pub.publish(msg)
            
            # Publish dummy camera info
            info_msg = CameraInfo()
            info_msg.header = msg.header
            info_msg.width = frame.shape[1]
            info_msg.height = frame.shape[0]
            info_msg.distortion_model = "plumb_bob"
            info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            
            # Approximate intrinsics for ~60 deg FOV
            fx = info_msg.width * 0.8 # Rough estimate
            fy = fx
            cx = info_msg.width / 2.0
            cy = info_msg.height / 2.0
            
            info_msg.k = [fx, 0.0, cx, 
                          0.0, fy, cy, 
                          0.0, 0.0, 1.0]
            info_msg.p = [fx, 0.0, cx, 0.0,
                          0.0, fy, cy, 0.0,
                          0.0, 0.0, 1.0, 0.0]
                          
            self.info_pub.publish(info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
