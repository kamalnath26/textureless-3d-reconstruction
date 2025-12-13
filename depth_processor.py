#!/usr/bin/env python3
"""
Depth Anything V1/V2/V3 Processor with Point Cloud Generation and ROS2 Publishing

Features:
- Support for Depth Anything V1, V2, V3 (default: V2)
- Input from folder/images or USB camera/video stream
- Configurable frame capture rate (1 fps, all frames, or custom percentage)
- Depth image output to folder
- Point cloud generation with camera intrinsics
- ROS2 topic publishing (depth images, point clouds)
- Options: images only, pointcloud only, or both

Usage Examples:
    # From images folder, output depth images only
    python depth_processor.py --input ./images --output ./output --mode images

    # From USB camera, 1 fps, output both depth and pointcloud
    python depth_processor.py --source camera --device 0 --fps-mode 1fps --mode both --pointcloud

    # From video, custom fps (50%), with ROS2 publishing
    python depth_processor.py --source video --video-path video.mp4 --fps-percent 50 --ros2 --ros2-freq 10

    # Using V3 model with metric depth
    python depth_processor.py --version v3 --input ./images --output ./output --metric --max-depth 80

Author: Generated for Kamalnath's robotics sensor fusion project
"""

import argparse
import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from threading import Thread, Event
from queue import Queue
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing torch and depth models
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch torchvision")

# Try importing Open3D for point cloud operations
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    logger.warning("Open3D not available. Install with: pip install open3d")

# Try importing ROS2
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
    from std_msgs.msg import Header
    from cv_bridge import CvBridge
    import sensor_msgs_py.point_cloud2 as pc2
    ROS2_AVAILABLE = True
except ImportError:
    logger.warning("ROS2 not available. ROS2 publishing will be disabled.")


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int
    depth_scale: float = 1.0  # Depth scale factor (for metric depth)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'CameraIntrinsics':
        """Load intrinsics from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(
            fx=data.get('fx', data.get('focal_length_x', 470.4)),
            fy=data.get('fy', data.get('focal_length_y', 470.4)),
            cx=data.get('cx', data.get('principal_point_x', data['width'] / 2)),
            cy=data.get('cy', data.get('principal_point_y', data['height'] / 2)),
            width=data['width'],
            height=data['height'],
            depth_scale=data.get('depth_scale', 1.0)
        )
    
    @classmethod
    def default(cls, width: int = 640, height: int = 480) -> 'CameraIntrinsics':
        """Create default intrinsics (typical webcam/RealSense values)"""
        return cls(
            fx=width * 0.8,  # Approximate focal length
            fy=width * 0.8,
            cx=width / 2,
            cy=height / 2,
            width=width,
            height=height
        )
    
    @classmethod
    def realsense_d455(cls) -> 'CameraIntrinsics':
        """Default RealSense D455 intrinsics (640x480)"""
        return cls(
            fx=382.193,
            fy=382.193,
            cx=320.819,
            cy=237.683,
            width=640,
            height=480,
            depth_scale=0.001  # 1mm per unit
        )
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


class DepthAnythingModel:
    """Wrapper for Depth Anything V1/V2/V3 models"""
    
    MODEL_CONFIGS = {
        'v1': {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        },
        'v2': {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
        },
        'v3': {
            'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
    }
    
    def __init__(
        self,
        version: str = 'v2',
        encoder: str = 'vitl',
        checkpoint_path: Optional[str] = None,
        metric: bool = False,
        max_depth: float = 20.0,
        dataset: str = 'hypersim',  # 'hypersim' for indoor, 'vkitti' for outdoor
        device: str = 'auto',
        input_size: int = 518,
        focal_length_ref: float = 300.0  # Reference focal length for V3 scaling
    ):
        """
        Initialize Depth Anything model
        
        Args:
            version: Model version ('v1', 'v2', 'v3')
            encoder: Encoder type ('vits', 'vitb', 'vitl', 'vitg')
            checkpoint_path: Path to model checkpoint
            metric: Use metric depth model
            max_depth: Maximum depth for metric models (20 indoor, 80 outdoor)
            dataset: Training dataset for metric models
            device: Device to run on ('auto', 'cuda', 'cpu', 'mps')
            input_size: Input size for inference
            focal_length_ref: Reference focal length for depth scaling (V3 uses 300.0)
        """
        self.version = version.lower()
        self.encoder = encoder
        self.metric = metric
        self.max_depth = max_depth
        self.dataset = dataset
        self.input_size = input_size
        self.focal_length_ref = focal_length_ref
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model = self.model.to(self.device).eval()
        
    def _load_model(self, checkpoint_path: Optional[str]):
        """Load the appropriate model based on version"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required but not available")
        
        config = self.MODEL_CONFIGS.get(self.version, {}).get(self.encoder)
        if config is None:
            raise ValueError(f"Invalid version/encoder combination: {self.version}/{self.encoder}")
        
        try:
            if self.version == 'v1':
                from depth_anything.dpt import DepthAnything
                model = DepthAnything(**config)
            elif self.version == 'v2':
                from depth_anything_v2.dpt import DepthAnythingV2
                if self.metric:
                    config = {**config, 'max_depth': self.max_depth}
                model = DepthAnythingV2(**config)
            elif self.version == 'v3':
                # V3 might use similar architecture or TensorRT
                # Try loading as V2 compatible or specific V3 implementation
                try:
                    from depth_anything_v3.dpt import DepthAnythingV3
                    config = {**config, 'max_depth': self.max_depth}
                    model = DepthAnythingV3(**config)
                except ImportError:
                    logger.warning("V3 specific module not found, trying V2 architecture")
                    from depth_anything_v2.dpt import DepthAnythingV2
                    config = {**config, 'max_depth': self.max_depth}
                    model = DepthAnythingV2(**config)
            else:
                raise ValueError(f"Unknown version: {self.version}")
                
        except ImportError as e:
            logger.error(f"Failed to import Depth Anything {self.version}: {e}")
            logger.info("Attempting to use Hugging Face transformers...")
            return self._load_from_transformers()
        
        # Load checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            # state_dict = torch.load(checkpoint_path, map_location='cpu')
            state_dict = torch.load(checkpoint_path, map_location='gpu')
            model.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning("No checkpoint provided or found, using uninitialized model")
            
        return model
    
    def _load_from_transformers(self):
        """Load model using Hugging Face transformers"""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        
        # Model mapping - using correct HF model names
        # Note: Metric models use different naming convention
        model_map = {
            ('v1', 'vits'): 'LiheYoung/depth-anything-small-hf',
            ('v1', 'vitb'): 'LiheYoung/depth-anything-base-hf',
            ('v1', 'vitl'): 'LiheYoung/depth-anything-large-hf',
            ('v2', 'vits'): 'depth-anything/Depth-Anything-V2-Small-hf',
            ('v2', 'vitb'): 'depth-anything/Depth-Anything-V2-Base-hf',
            ('v2', 'vitl'): 'depth-anything/Depth-Anything-V2-Large-hf',
        }
        
        # Metric models have different naming and may require authentication
        if self.metric:
            dataset_name = "Indoor" if self.dataset == "hypersim" else "Outdoor"
            model_map.update({
                ('v2', 'vits'): f'depth-anything/Depth-Anything-V2-Metric-{dataset_name}-Small-hf',
                ('v2', 'vitb'): f'depth-anything/Depth-Anything-V2-Metric-{dataset_name}-Base-hf',
                ('v2', 'vitl'): f'depth-anything/Depth-Anything-V2-Metric-{dataset_name}-Large-hf',
            })
        
        model_name = model_map.get((self.version, self.encoder))
        if model_name is None:
            raise ValueError(f"No Hugging Face model for {self.version}/{self.encoder}")
        
        logger.info(f"Loading from Hugging Face: {model_name}")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForDepthEstimation.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            logger.info("Trying non-metric model as fallback...")
            # Fallback to non-metric model
            fallback_map = {
                ('v2', 'vits'): 'depth-anything/Depth-Anything-V2-Small-hf',
                ('v2', 'vitb'): 'depth-anything/Depth-Anything-V2-Base-hf',
                ('v2', 'vitl'): 'depth-anything/Depth-Anything-V2-Large-hf',
            }
            model_name = fallback_map.get((self.version, self.encoder))
            if model_name:
                logger.info(f"Loading fallback model: {model_name}")
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                model = AutoModelForDepthEstimation.from_pretrained(model_name)
                logger.warning("Using relative depth model - output will NOT be in meters!")
            else:
                raise
        
        self._use_transformers = True
        return model
    
    def infer(self, image: np.ndarray, intrinsics: Optional[CameraIntrinsics] = None) -> np.ndarray:
        """
        Run depth inference on an image
        
        Args:
            image: BGR image (OpenCV format)
            intrinsics: Camera intrinsics for focal length scaling
            
        Returns:
            Depth map (HxW float32 array in meters if metric, otherwise relative)
        """
        if hasattr(self, '_use_transformers') and self._use_transformers:
            return self._infer_transformers(image)
        
        # Standard inference
        with torch.no_grad():
            depth = self.model.infer_image(image, self.input_size)
        
        # Apply focal length scaling for V3 (and optionally V2 metric)
        if self.version == 'v3' and intrinsics is not None:
            focal_pixels = (intrinsics.fx + intrinsics.fy) / 2
            scale_factor = focal_pixels / self.focal_length_ref
            depth = depth * scale_factor
        
        return depth.astype(np.float32)
    
    def _infer_transformers(self, image: np.ndarray) -> np.ndarray:
        """Inference using transformers pipeline"""
        from PIL import Image as PILImage
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        
        depth = prediction.squeeze().cpu().numpy()
        return depth.astype(np.float32)


class PointCloudGenerator:
    """Generate point clouds from depth images using camera intrinsics"""
    
    def __init__(self, intrinsics: CameraIntrinsics, downsample_factor: int = 1):
        """
        Initialize point cloud generator
        
        Args:
            intrinsics: Camera intrinsic parameters
            downsample_factor: Downsample point cloud by this factor
        """
        self.intrinsics = intrinsics
        self.downsample = downsample_factor
        
        # Pre-compute pixel coordinate grids
        self._precompute_grids()
    
    def _precompute_grids(self):
        """Pre-compute pixel coordinate grids for efficiency"""
        h, w = self.intrinsics.height, self.intrinsics.width
        
        # Create meshgrid of pixel coordinates
        u = np.arange(0, w, self.downsample)
        v = np.arange(0, h, self.downsample)
        self.u_grid, self.v_grid = np.meshgrid(u, v)
        
        # Normalize pixel coordinates
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        self.x_norm = (self.u_grid - self.intrinsics.cx) / self.intrinsics.fx
        self.y_norm = (self.v_grid - self.intrinsics.cy) / self.intrinsics.fy
    
    def generate(
        self,
        depth: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        max_depth: float = 100.0,
        min_depth: float = 0.1
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate point cloud from depth image
        
        Based on Depth Anything V3 ROS2 implementation:
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        
        Args:
            depth: Depth image (HxW float32 in meters)
            rgb: Optional RGB image for coloring points
            max_depth: Maximum valid depth
            min_depth: Minimum valid depth
            
        Returns:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors (0-1) or None
        """
        # Downsample depth if needed
        if self.downsample > 1:
            depth = depth[::self.downsample, ::self.downsample]
        
        # Create valid depth mask
        valid_mask = (depth > min_depth) & (depth < max_depth) & np.isfinite(depth)
        
        # Compute 3D coordinates
        z = depth
        x = self.x_norm * z
        y = self.y_norm * z
        
        # Stack and reshape
        points = np.stack([x, y, z], axis=-1)
        points = points[valid_mask]
        
        # Handle colors
        colors = None
        if rgb is not None:
            if self.downsample > 1:
                rgb = rgb[::self.downsample, ::self.downsample]
            colors = rgb[valid_mask].astype(np.float32) / 255.0
            # Convert BGR to RGB if needed
            if colors.shape[1] == 3:
                colors = colors[:, ::-1]  # BGR to RGB
        
        return points.astype(np.float32), colors
    
    def save_ply(
        self,
        filepath: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None
    ):
        """Save point cloud to PLY file using Open3D"""
        if not O3D_AVAILABLE:
            logger.error("Open3D required for saving PLY files")
            return
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(filepath, pcd)
        logger.debug(f"Saved point cloud to {filepath}")
    
    def save_pcd(
        self,
        filepath: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None
    ):
        """Save point cloud to PCD file"""
        self.save_ply(filepath.replace('.ply', '.pcd'), points, colors)


class ImageSource:
    """Base class for image sources"""
    
    def __init__(self):
        self.intrinsics: Optional[CameraIntrinsics] = None
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[np.ndarray, float, str]:
        """Return (image, timestamp, identifier)"""
        raise NotImplementedError
    
    def close(self):
        pass


class FolderSource(ImageSource):
    """Load images from a folder"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, folder_path: str, intrinsics_path: Optional[str] = None):
        super().__init__()
        self.folder = Path(folder_path)
        
        if not self.folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Get sorted list of images
        self.images = sorted([
            f for f in self.folder.iterdir()
            if f.suffix.lower() in self.SUPPORTED_FORMATS
        ])
        
        if not self.images:
            raise ValueError(f"No images found in {folder_path}")
        
        logger.info(f"Found {len(self.images)} images in {folder_path}")
        self.index = 0
        
        # Load intrinsics
        if intrinsics_path:
            self.intrinsics = CameraIntrinsics.from_json(intrinsics_path)
        else:
            # Try to infer from first image
            img = cv2.imread(str(self.images[0]))
            h, w = img.shape[:2]
            self.intrinsics = CameraIntrinsics.default(w, h)
    
    def __len__(self):
        return len(self.images)
    
    def __next__(self) -> Tuple[np.ndarray, float, str]:
        if self.index >= len(self.images):
            raise StopIteration
        
        img_path = self.images[self.index]
        self.index += 1
        
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to load image: {img_path}")
            return self.__next__()
        
        timestamp = time.time()
        return img, timestamp, img_path.stem


class CameraSource(ImageSource):
    """Capture from USB camera or video device"""
    
    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        fps_mode: str = '1fps',  # '1fps', 'all', or 'custom'
        fps_percent: float = 100.0,
        intrinsics_path: Optional[str] = None
    ):
        super().__init__()
        self.device = device
        self.fps_mode = fps_mode
        self.fps_percent = fps_percent
        
        # Open camera
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {device}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Get actual resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        logger.info(f"Camera opened: {self.width}x{self.height} @ {self.source_fps} fps")
        
        # Load intrinsics
        if intrinsics_path:
            self.intrinsics = CameraIntrinsics.from_json(intrinsics_path)
        else:
            self.intrinsics = CameraIntrinsics.default(self.width, self.height)
        
        # Frame timing
        self.last_capture_time = 0
        self.frame_count = 0
        self._calculate_capture_interval()
    
    def _calculate_capture_interval(self):
        """Calculate how often to capture frames"""
        if self.fps_mode == '1fps':
            self.capture_interval = 1.0
        elif self.fps_mode == 'all':
            self.capture_interval = 0
        else:  # custom percentage
            target_fps = self.source_fps * (self.fps_percent / 100.0)
            self.capture_interval = 1.0 / target_fps if target_fps > 0 else 0
    
    def __next__(self) -> Tuple[np.ndarray, float, str]:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration
            
            current_time = time.time()
            
            # Check if we should capture this frame
            if self.capture_interval == 0 or \
               (current_time - self.last_capture_time) >= self.capture_interval:
                self.last_capture_time = current_time
                self.frame_count += 1
                return frame, current_time, f"frame_{self.frame_count:06d}"
    
    def close(self):
        if self.cap:
            self.cap.release()


class VideoSource(ImageSource):
    """Load frames from a video file"""
    
    def __init__(
        self,
        video_path: str,
        fps_mode: str = '1fps',
        fps_percent: float = 100.0,
        intrinsics_path: Optional[str] = None
    ):
        super().__init__()
        self.video_path = video_path
        self.fps_mode = fps_mode
        self.fps_percent = fps_percent
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        logger.info(f"Video opened: {self.width}x{self.height}, {self.total_frames} frames @ {self.source_fps} fps")
        
        # Load intrinsics
        if intrinsics_path:
            self.intrinsics = CameraIntrinsics.from_json(intrinsics_path)
        else:
            self.intrinsics = CameraIntrinsics.default(self.width, self.height)
        
        # Calculate frame skip
        self._calculate_frame_skip()
        self.frame_index = 0
    
    def _calculate_frame_skip(self):
        """Calculate how many frames to skip"""
        if self.fps_mode == '1fps':
            self.frame_skip = int(self.source_fps)
        elif self.fps_mode == 'all':
            self.frame_skip = 1
        else:  # custom percentage
            self.frame_skip = max(1, int(100.0 / self.fps_percent))
    
    def __next__(self) -> Tuple[np.ndarray, float, str]:
        while self.frame_index < self.total_frames:
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            ret, frame = self.cap.read()
            
            if not ret:
                self.frame_index += self.frame_skip
                continue
            
            timestamp = self.frame_index / self.source_fps
            identifier = f"frame_{self.frame_index:06d}"
            self.frame_index += self.frame_skip
            
            return frame, timestamp, identifier
        
        raise StopIteration
    
    def close(self):
        if self.cap:
            self.cap.release()


class ROS2DepthPublisher(Node):
    """ROS2 node for publishing depth images and point clouds"""
    
    def __init__(
        self,
        node_name: str = 'depth_anything_processor',
        publish_depth: bool = True,
        publish_pointcloud: bool = True,
        depth_topic: str = '/depth_anything/depth_image',
        pointcloud_topic: str = '/depth_anything/points',
        camera_info_topic: str = '/depth_anything/camera_info',
        frame_id: str = 'camera_depth_optical_frame',
        publish_rate: float = 10.0
    ):
        super().__init__(node_name)
        
        self.frame_id = frame_id
        self.bridge = CvBridge()
        self.publish_rate = publish_rate
        
        # Create publishers
        self.depth_pub = None
        self.pc_pub = None
        self.camera_info_pub = None
        
        if publish_depth:
            self.depth_pub = self.create_publisher(Image, depth_topic, 10)
            logger.info(f"Publishing depth to: {depth_topic}")
        
        if publish_pointcloud:
            self.pc_pub = self.create_publisher(PointCloud2, pointcloud_topic, 10)
            logger.info(f"Publishing point cloud to: {pointcloud_topic}")
        
        self.camera_info_pub = self.create_publisher(CameraInfo, camera_info_topic, 10)
        
        # Rate limiter
        self.last_publish_time = 0
        self.min_publish_interval = 1.0 / publish_rate
    
    def _create_header(self, timestamp: float) -> Header:
        """Create a ROS2 header with timestamp"""
        header = Header()
        header.stamp.sec = int(timestamp)
        header.stamp.nanosec = int((timestamp % 1) * 1e9)
        header.frame_id = self.frame_id
        return header
    
    def publish_depth_image(self, depth: np.ndarray, timestamp: float):
        """Publish depth image as 32FC1"""
        if self.depth_pub is None:
            return
        
        # Ensure correct dtype
        depth_32f = depth.astype(np.float32)
        
        # Convert to ROS message
        msg = self.bridge.cv2_to_imgmsg(depth_32f, encoding='32FC1')
        msg.header = self._create_header(timestamp)
        
        self.depth_pub.publish(msg)
    
    def publish_pointcloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        timestamp: float
    ):
        """Publish point cloud as PointCloud2"""
        if self.pc_pub is None:
            return
        
        header = self._create_header(timestamp)
        
        # Create fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        if colors is not None:
            fields.append(
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
            )
            # Pack RGB into float32
            rgb_packed = np.zeros(len(points), dtype=np.float32)
            for i in range(len(points)):
                r, g, b = (colors[i] * 255).astype(np.uint8)
                rgb_packed[i] = np.frombuffer(
                    np.array([b, g, r, 0], dtype=np.uint8).tobytes(),
                    dtype=np.float32
                )[0]
            
            cloud_data = np.column_stack([points, rgb_packed])
        else:
            cloud_data = points
        
        # Create message
        msg = pc2.create_cloud(header, fields, cloud_data)
        self.pc_pub.publish(msg)
    
    def publish_camera_info(self, intrinsics: CameraIntrinsics, timestamp: float):
        """Publish camera info message"""
        msg = CameraInfo()
        msg.header = self._create_header(timestamp)
        msg.width = intrinsics.width
        msg.height = intrinsics.height
        
        # Intrinsic matrix K
        msg.k = [
            intrinsics.fx, 0.0, intrinsics.cx,
            0.0, intrinsics.fy, intrinsics.cy,
            0.0, 0.0, 1.0
        ]
        
        # Distortion (assume none)
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.distortion_model = 'plumb_bob'
        
        self.camera_info_pub.publish(msg)
    
    def should_publish(self) -> bool:
        """Check if enough time has passed since last publish"""
        current_time = time.time()
        if current_time - self.last_publish_time >= self.min_publish_interval:
            self.last_publish_time = current_time
            return True
        return False


class DepthProcessor:
    """Main processor class that ties everything together"""
    
    def __init__(
        self,
        model: DepthAnythingModel,
        source: ImageSource,
        output_dir: str,
        mode: str = 'both',  # 'images', 'pointcloud', 'both'
        enable_ros2: bool = False,
        ros2_freq: float = 10.0,
        pointcloud_downsample: int = 1,
        max_depth: float = 100.0,
        min_depth: float = 0.1,
        colormap: int = cv2.COLORMAP_JET,
        save_raw_depth: bool = True
    ):
        self.model = model
        self.source = source
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.enable_ros2 = enable_ros2 and ROS2_AVAILABLE
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.colormap = colormap
        self.save_raw_depth = save_raw_depth
        
        # Create output directories
        self.depth_dir = self.output_dir / 'depth_images'
        self.pc_dir = self.output_dir / 'pointclouds'
        self.vis_dir = self.output_dir / 'visualizations'
        
        if mode in ['images', 'both']:
            self.depth_dir.mkdir(parents=True, exist_ok=True)
            self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        if mode in ['pointcloud', 'both']:
            self.pc_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize point cloud generator
        self.pc_generator = PointCloudGenerator(
            source.intrinsics,
            downsample_factor=pointcloud_downsample
        )
        
        # Initialize ROS2 if enabled
        self.ros2_node = None
        if self.enable_ros2:
            rclpy.init()
            self.ros2_node = ROS2DepthPublisher(
                publish_depth=mode in ['images', 'both'],
                publish_pointcloud=mode in ['pointcloud', 'both'],
                publish_rate=ros2_freq
            )
    
    def process(self, show_preview: bool = False):
        """Process all images from source"""
        logger.info(f"Starting processing with mode: {self.mode}")
        
        processed_count = 0
        start_time = time.time()
        
        try:
            for image, timestamp, identifier in self.source:
                # Run depth inference
                depth = self.model.infer(image, self.source.intrinsics)
                
                # Save/publish depth images
                if self.mode in ['images', 'both']:
                    self._save_depth(depth, identifier)
                
                # Generate and save/publish point cloud
                points, colors = None, None
                if self.mode in ['pointcloud', 'both']:
                    points, colors = self.pc_generator.generate(
                        depth, image, self.max_depth, self.min_depth
                    )
                    self._save_pointcloud(points, colors, identifier)
                
                # ROS2 publishing
                if self.enable_ros2 and self.ros2_node.should_publish():
                    self.ros2_node.publish_camera_info(self.source.intrinsics, timestamp)
                    
                    if self.mode in ['images', 'both']:
                        self.ros2_node.publish_depth_image(depth, timestamp)
                    
                    if self.mode in ['pointcloud', 'both'] and points is not None:
                        self.ros2_node.publish_pointcloud(points, colors, timestamp)
                    
                    rclpy.spin_once(self.ros2_node, timeout_sec=0)
                
                # Preview
                if show_preview:
                    self._show_preview(image, depth, identifier)
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = processed_count / elapsed
                    logger.info(f"Processed {processed_count} frames ({fps:.1f} fps)")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        
        finally:
            elapsed = time.time() - start_time
            logger.info(f"Processed {processed_count} frames in {elapsed:.1f}s ({processed_count/elapsed:.1f} fps)")
            self.cleanup()
    
    def _save_depth(self, depth: np.ndarray, identifier: str):
        """Save depth image"""
        # Save raw depth as numpy or EXR
        if self.save_raw_depth:
            np.save(self.depth_dir / f"{identifier}_depth.npy", depth)
        
        # Save visualization
        depth_normalized = np.clip(depth / self.max_depth, 0, 1)
        depth_colored = cv2.applyColorMap(
            (depth_normalized * 255).astype(np.uint8),
            self.colormap
        )
        cv2.imwrite(str(self.vis_dir / f"{identifier}_depth_vis.png"), depth_colored)
        
        # Save 16-bit depth image (in millimeters)
        depth_mm = (depth * 1000).astype(np.uint16)
        cv2.imwrite(str(self.depth_dir / f"{identifier}_depth.png"), depth_mm)
    
    def _save_pointcloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        identifier: str
    ):
        """Save point cloud"""
        if points is None or len(points) == 0:
            return
        
        filepath = str(self.pc_dir / f"{identifier}.ply")
        self.pc_generator.save_ply(filepath, points, colors)
    
    def _show_preview(self, image: np.ndarray, depth: np.ndarray, identifier: str):
        """Show preview window"""
        depth_normalized = np.clip(depth / self.max_depth, 0, 1)
        depth_colored = cv2.applyColorMap(
            (depth_normalized * 255).astype(np.uint8),
            self.colormap
        )
        
        # Resize if needed
        h, w = image.shape[:2]
        if w > 640:
            scale = 640 / w
            image = cv2.resize(image, None, fx=scale, fy=scale)
            depth_colored = cv2.resize(depth_colored, None, fx=scale, fy=scale)
        
        combined = np.hstack([image, depth_colored])
        cv2.imshow(f'Depth Anything - {identifier}', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt
    
    def cleanup(self):
        """Cleanup resources"""
        self.source.close()
        cv2.destroyAllWindows()
        
        if self.ros2_node:
            self.ros2_node.destroy_node()
            rclpy.shutdown()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Depth Anything Processor with Point Cloud Generation and ROS2 Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model settings
    model_group = parser.add_argument_group('Model Settings')
    model_group.add_argument('--version', type=str, default='v2', choices=['v1', 'v2', 'v3'],
                            help='Depth Anything version (default: v2)')
    model_group.add_argument('--encoder', type=str, default='vitl',
                            choices=['vits', 'vitb', 'vitl', 'vitg', 'large'],
                            help='Encoder size (default: vitl)')
    model_group.add_argument('--checkpoint', type=str, default=None,
                            help='Path to model checkpoint')
    model_group.add_argument('--metric', action='store_true',
                            help='Use metric depth model')
    model_group.add_argument('--max_depth', type=float, default=20.0,
                            help='Maximum depth for metric models (20 indoor, 80 outdoor)')
    model_group.add_argument('--dataset', type=str, default='hypersim',
                            choices=['hypersim', 'vkitti'],
                            help='Training dataset for metric models')
    model_group.add_argument('--input-size', type=int, default=518,
                            help='Input size for model inference')
    model_group.add_argument('--device', type=str, default='auto',
                            choices=['auto', 'cuda', 'cpu', 'mps'],
                            help='Device for inference')
    
    # Input settings
    input_group = parser.add_argument_group('Input Settings')
    input_group.add_argument('--source', type=str, default='folder',
                            choices=['folder', 'camera', 'video'],
                            help='Input source type')
    input_group.add_argument('--input', type=str, default='./images',
                            help='Input folder path (for folder source)')
    input_group.add_argument('--video-path', type=str,
                            help='Video file path (for video source)')
    input_group.add_argument('--device-id', type=int, default=0,
                            help='Camera device ID (for camera source)')
    input_group.add_argument('--width', type=int, default=640,
                            help='Camera/video width')
    input_group.add_argument('--height', type=int, default=480,
                            help='Camera/video height')
    input_group.add_argument('--fps-mode', type=str, default='1fps',
                            choices=['1fps', 'all', 'custom'],
                            help='Frame capture mode')
    input_group.add_argument('--fps-percent', type=float, default=100.0,
                            help='FPS percentage for custom mode (1-100)')
    input_group.add_argument('--intrinsics', type=str,
                            help='Path to camera intrinsics JSON file')
    
    # Output settings
    output_group = parser.add_argument_group('Output Settings')
    output_group.add_argument('--output', type=str, default='./output',
                             help='Output directory')
    output_group.add_argument('--mode', type=str, default='both',
                             choices=['images', 'pointcloud', 'both'],
                             help='Output mode')
    output_group.add_argument('--pointcloud-downsample', type=int, default=1,
                             help='Point cloud downsampling factor')
    output_group.add_argument('--min_depth', type=float, default=0.1,
                             help='Minimum valid depth (meters)')
    output_group.add_argument('--colormap', type=str, default='jet',
                             choices=['jet', 'magma', 'inferno', 'viridis', 'plasma', 'turbo'],
                             help='Depth visualization colormap')
    output_group.add_argument('--no-raw-depth', action='store_true',
                             help='Do not save raw depth numpy files')
    
    # ROS2 settings
    ros2_group = parser.add_argument_group('ROS2 Settings')
    ros2_group.add_argument('--ros2', action='store_true',
                           help='Enable ROS2 topic publishing')
    ros2_group.add_argument('--ros2-freq', type=float, default=10.0,
                           help='ROS2 publish frequency (Hz)')
    ros2_group.add_argument('--depth-topic', type=str, default='/depth_anything/depth_image',
                           help='ROS2 depth image topic')
    ros2_group.add_argument('--pc-topic', type=str, default='/depth_anything/points',
                           help='ROS2 point cloud topic')
    ros2_group.add_argument('--frame-id', type=str, default='camera_depth_optical_frame',
                           help='ROS2 frame ID')
    
    # Other settings
    parser.add_argument('--preview', action='store_true',
                       help='Show preview window')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    return parser.parse_args()


def get_colormap(name: str) -> int:
    """Get OpenCV colormap from name"""
    colormaps = {
        'jet': cv2.COLORMAP_JET,
        'magma': cv2.COLORMAP_MAGMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'turbo': cv2.COLORMAP_TURBO,
    }
    return colormaps.get(name.lower(), cv2.COLORMAP_JET)


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check dependencies
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required. Install with: pip install torch torchvision")
        sys.exit(1)
    
    if args.mode in ['pointcloud', 'both'] and not O3D_AVAILABLE:
        logger.error("Open3D is required for point cloud generation. Install with: pip install open3d")
        sys.exit(1)
    
    if args.ros2 and not ROS2_AVAILABLE:
        logger.error("ROS2 is required for topic publishing but not available")
        sys.exit(1)
    
    # Initialize model
    logger.info(f"Loading Depth Anything {args.version.upper()} with {args.encoder} encoder...")
    model = DepthAnythingModel(
        version=args.version,
        encoder=args.encoder,
        checkpoint_path=args.checkpoint,
        metric=args.metric,
        max_depth=args.max_depth,
        dataset=args.dataset,
        device=args.device,
        input_size=args.input_size
    )
    
    # Initialize source
    if args.source == 'folder':
        source = FolderSource(args.input, args.intrinsics)
    elif args.source == 'camera':
        source = CameraSource(
            device=args.device_id,
            width=args.width,
            height=args.height,
            fps_mode=args.fps_mode,
            fps_percent=args.fps_percent,
            intrinsics_path=args.intrinsics
        )
    elif args.source == 'video':
        if not args.video_path:
            logger.error("--video-path required for video source")
            sys.exit(1)
        source = VideoSource(
            video_path=args.video_path,
            fps_mode=args.fps_mode,
            fps_percent=args.fps_percent,
            intrinsics_path=args.intrinsics
        )
    
    # Initialize processor
    processor = DepthProcessor(
        model=model,
        source=source,
        output_dir=args.output,
        mode=args.mode,
        enable_ros2=args.ros2,
        ros2_freq=args.ros2_freq,
        pointcloud_downsample=args.pointcloud_downsample,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        colormap=get_colormap(args.colormap),
        save_raw_depth=not args.no_raw_depth
    )
    
    # Run processing
    processor.process(show_preview=args.preview)


if __name__ == '__main__':
    main()