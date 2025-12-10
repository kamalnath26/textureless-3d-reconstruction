#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import sqlite3
import numpy as np
import time
import os

class DBPlayerNode(Node):
    def __init__(self):
        super().__init__('db_player_node')
        
        self.declare_parameter('db_path', '')
        self.declare_parameter('framerate', 30.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('frame_id', 'camera_optical_frame')

        self.db_path = self.get_parameter('db_path').get_parameter_value().string_value
        self.framerate = self.get_parameter('framerate').get_parameter_value().double_value
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        if not self.db_path or not os.path.exists(self.db_path):
            self.get_logger().error(f"Database path not provided or does not exist: {self.db_path}")
            raise ValueError("Invalid database path")

        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.bridge = CvBridge()
        
        self.timer_period = 1.0 / self.framerate
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.conn = None
        self.cursor = None
        self.ids = []
        self.current_index = 0
        
        self.connect_db()
        self.load_ids()
        self.stored_camera_info = self.get_calibration()
        
        self.get_logger().info(f"DB Player Node started. Playing from {self.db_path} at {self.framerate} FPS. Total frames: {len(self.ids)}")

    def connect_db(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        except Exception as e:
            self.get_logger().error(f"Failed to connect to database: {e}")
            raise

    def load_ids(self):
        try:
            # Get all node IDs that have image data
            query = """
                SELECT Node.id 
                FROM Node 
                JOIN Data ON Node.id = Data.id 
                WHERE Data.image IS NOT NULL 
                ORDER BY Node.id ASC
            """
            self.cursor.execute(query)
            self.ids = [row[0] for row in self.cursor.fetchall()]
            
            if not self.ids:
                self.get_logger().warn("No images found in the database!")
        except Exception as e:
            self.get_logger().error(f"Failed to load IDs: {e}")

    def get_frame_data(self, node_id):
        try:
            query = "SELECT image FROM Data WHERE id = ?"
            self.cursor.execute(query, (node_id,))
            data = self.cursor.fetchone()
            
            if data and data[0]:
                image_data = data[0]
                nparr = np.frombuffer(image_data, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return cv_image
            return None
        except Exception as e:
            self.get_logger().error(f"Error reading frame {node_id}: {e}")
            return None

    def get_calibration(self):
        try:
            query = "SELECT calibration FROM Data WHERE calibration IS NOT NULL LIMIT 1"
            self.cursor.execute(query)
            data = self.cursor.fetchone()
            
            if data and data[0]:
                calib_data = data[0]
                # Parse based on our inspection
                # Ints at offset 0
                ints = np.frombuffer(calib_data, dtype=np.int32)
                width = ints[4]
                height = ints[5]
                
                # Doubles at offset 44 (K matrix)
                # [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                doubles = np.frombuffer(calib_data, dtype=np.float64, offset=44)
                fx = doubles[0]
                cx = doubles[2]
                fy = doubles[4]
                cy = doubles[5]
                
                self.get_logger().info(f"Loaded calibration: {width}x{height} fx={fx} fy={fy} cx={cx} cy={cy}")
                
                info_msg = CameraInfo()
                info_msg.width = int(width)
                info_msg.height = int(height)
                info_msg.distortion_model = "plumb_bob"
                info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
                info_msg.k = [fx, 0.0, cx, 
                              0.0, fy, cy, 
                              0.0, 0.0, 1.0]
                info_msg.p = [fx, 0.0, cx, 0.0,
                              0.0, fy, cy, 0.0,
                              0.0, 0.0, 1.0, 0.0]
                
                return info_msg
            else:
                self.get_logger().warn("No calibration found in DB, using default.")
                return None
        except Exception as e:
            self.get_logger().error(f"Error reading calibration: {e}")
            return None

    def timer_callback(self):
        if not self.ids:
            return

        if self.current_index >= len(self.ids):
            if self.loop:
                self.current_index = 0
                self.get_logger().info("Looping playback...")
            else:
                self.get_logger().info("End of database reached.")
                self.timer.cancel()
                return

        node_id = self.ids[self.current_index]
        cv_image = self.get_frame_data(node_id)

        if cv_image is not None:
            msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            self.publisher_.publish(msg)
            
            # Publish Camera Info
            if self.stored_camera_info:
                # Create a deep copy to avoid modifying the stored one permanently
                import copy
                info_msg = copy.deepcopy(self.stored_camera_info)
                info_msg.header = msg.header
                
                # Check if image size matches calibration size
                if cv_image.shape[1] != info_msg.width or cv_image.shape[0] != info_msg.height:
                    # Scale intrinsics if necessary
                    scale_x = cv_image.shape[1] / info_msg.width
                    scale_y = cv_image.shape[0] / info_msg.height
                    
                    info_msg.width = cv_image.shape[1]
                    info_msg.height = cv_image.shape[0]
                    info_msg.k[0] *= scale_x # fx
                    info_msg.k[2] *= scale_x # cx
                    info_msg.k[4] *= scale_y # fy
                    info_msg.k[5] *= scale_y # cy
                    
                    info_msg.p[0] *= scale_x # fx
                    info_msg.p[2] *= scale_x # cx
                    info_msg.p[5] *= scale_y # fy
                    info_msg.p[6] *= scale_y # cy
                
                self.info_pub.publish(info_msg)
            else:
                # Fallback to approximate
                info_msg = CameraInfo()
                info_msg.header = msg.header
                info_msg.width = cv_image.shape[1]
                info_msg.height = cv_image.shape[0]
                info_msg.distortion_model = "plumb_bob"
                info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0] 
                
                fx = info_msg.width * 0.8 
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
            
        self.current_index += 1

    def destroy_node(self):
        if self.conn:
            self.conn.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = DBPlayerNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
