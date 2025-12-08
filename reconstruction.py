"""
Monocular 3D Reconstruction System
Handles both image folders and live camera input
Designed to work with textureless surfaces using depth estimation + geometric fusion

Requirements:
pip install opencv-python numpy torch torchvision open3d transformers pillow

For SLAM-like operation, also consider:
pip install pyrealsense2  # if using Intel RealSense for comparison
"""

import cv2
import numpy as np
import torch
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
import threading
from queue import Queue

# ============================================================================
# DEPTH ESTIMATION MODULE (Handles textureless surfaces)
# ============================================================================

class DepthEstimator:
    """
    Uses learning-based monocular depth estimation.
    These models handle textureless surfaces by learning geometric priors.
    """
    
    def __init__(self, model_type: str = "depth_anything"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load depth estimation model"""
        
        if self.model_type == "depth_anything":
            # Depth Anything - excellent for textureless surfaces
            from transformers import pipeline
            self.pipe = pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                device=0 if torch.cuda.is_available() else -1
            )
            
        elif self.model_type == "midas":
            # MiDaS - robust general-purpose depth
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
            
        print(f"Loaded {self.model_type} on {self.device}")
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a single RGB image.
        Returns relative depth map (higher = farther).
        """
        original_h, original_w = image.shape[:2]
        
        if self.model_type == "depth_anything":
            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result = self.pipe(pil_image)
            depth = np.array(result["depth"])
            # Ensure depth matches original input dimensions
            if depth.shape[0] != original_h or depth.shape[1] != original_w:
                depth = cv2.resize(depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            return depth
            
        elif self.model_type == "midas":
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img_rgb).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            return prediction.cpu().numpy()


# ============================================================================
# FEATURE DETECTION FOR TEXTURELESS SURFACES
# ============================================================================

class HybridFeatureDetector:
    """
    Combines point features with line/edge features for textureless surfaces.
    """
    
    def __init__(self):
        # Point features (for textured regions)
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.sift = cv2.SIFT_create(nfeatures=1000)
        
        # Line segment detector (for textureless regions with edges)
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        
        # For matching
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.flann_matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )
    
    def detect_features(self, image: np.ndarray) -> dict:
        """Detect both point and line features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Point features
        orb_kp, orb_desc = self.orb.detectAndCompute(gray, None)
        sift_kp, sift_desc = self.sift.detectAndCompute(gray, None)
        
        # Line features (crucial for textureless surfaces)
        lines, width, prec, nfa = self.lsd.detect(gray)
        
        # Edge-based features using Canny
        edges = cv2.Canny(gray, 50, 150)
        
        return {
            'orb': (orb_kp, orb_desc),
            'sift': (sift_kp, sift_desc),
            'lines': lines,
            'edges': edges,
            'gray': gray
        }
    
    def match_features(self, feat1: dict, feat2: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Match features between two frames"""
        pts1, pts2 = [], []
        
        # Match ORB features
        if feat1['orb'][1] is not None and feat2['orb'][1] is not None:
            matches = self.bf_matcher.match(feat1['orb'][1], feat2['orb'][1])
            matches = sorted(matches, key=lambda x: x.distance)[:100]
            
            for m in matches:
                pts1.append(feat1['orb'][0][m.queryIdx].pt)
                pts2.append(feat2['orb'][0][m.trainIdx].pt)
        
        # Match SIFT features (better for some textures)
        if feat1['sift'][1] is not None and feat2['sift'][1] is not None:
            matches = self.flann_matcher.knnMatch(
                feat1['sift'][1].astype(np.float32),
                feat2['sift'][1].astype(np.float32),
                k=2
            )
            # Ratio test
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    pts1.append(feat1['sift'][0][m.queryIdx].pt)
                    pts2.append(feat2['sift'][0][m.trainIdx].pt)
        
        return np.array(pts1), np.array(pts2)


# ============================================================================
# PLANE DETECTION (Key for textureless surfaces)
# ============================================================================

class PlaneDetector:
    """
    Detect planes in depth maps - essential for textureless surfaces
    like walls, floors, tables.
    """
    
    @staticmethod
    def detect_planes(depth: np.ndarray, 
                      num_planes: int = 3,
                      threshold: float = 0.02) -> List[dict]:
        """
        Detect dominant planes using RANSAC on 3D points.
        """
        h, w = depth.shape
        
        # Create point cloud from depth
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Normalize coordinates
        x_norm = (x_coords - w/2) / w
        y_norm = (y_coords - h/2) / h
        z_norm = depth / depth.max()
        
        # Stack into points
        points = np.stack([x_norm, y_norm, z_norm], axis=-1)
        points_flat = points.reshape(-1, 3)
        
        # Remove invalid points
        valid_mask = ~np.isnan(points_flat).any(axis=1)
        valid_points = points_flat[valid_mask]
        
        planes = []
        remaining_points = valid_points.copy()
        
        for _ in range(num_planes):
            if len(remaining_points) < 100:
                break
                
            # RANSAC plane fitting
            best_plane = None
            best_inliers = 0
            
            for _ in range(100):  # RANSAC iterations
                # Sample 3 points
                idx = np.random.choice(len(remaining_points), 3, replace=False)
                p1, p2, p3 = remaining_points[idx]
                
                # Compute plane normal
                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)
                
                if np.linalg.norm(normal) < 1e-6:
                    continue
                    
                normal = normal / np.linalg.norm(normal)
                d = -np.dot(normal, p1)
                
                # Count inliers
                distances = np.abs(np.dot(remaining_points, normal) + d)
                inliers = np.sum(distances < threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_plane = (normal, d, distances < threshold)
            
            if best_plane is not None and best_inliers > 50:
                normal, d, inlier_mask = best_plane
                planes.append({
                    'normal': normal,
                    'd': d,
                    'inlier_count': best_inliers
                })
                remaining_points = remaining_points[~inlier_mask]
        
        return planes


# ============================================================================
# VISUAL ODOMETRY / POSE ESTIMATION
# ============================================================================

class VisualOdometry:
    """
    Estimate camera motion between frames.
    """
    
    def __init__(self, camera_matrix: np.ndarray):
        self.K = camera_matrix
        self.feature_detector = HybridFeatureDetector()
        self.prev_frame = None
        self.prev_features = None
        self.poses = [np.eye(4)]  # Start at identity
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a new frame and estimate pose.
        Returns 4x4 transformation matrix.
        """
        features = self.feature_detector.detect_features(frame)
        
        if self.prev_features is None:
            self.prev_frame = frame
            self.prev_features = features
            return self.poses[-1]
        
        # Match features
        pts1, pts2 = self.feature_detector.match_features(
            self.prev_features, features
        )
        
        if len(pts1) < 8:
            print("Warning: Not enough matches")
            self.prev_frame = frame
            self.prev_features = features
            return self.poses[-1]
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        
        # Build transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        
        # Accumulate pose
        new_pose = self.poses[-1] @ T
        self.poses.append(new_pose)
        
        self.prev_frame = frame
        self.prev_features = features
        
        return new_pose


# ============================================================================
# POINT CLOUD GENERATION
# ============================================================================

class PointCloudGenerator:
    """
    Generate 3D point clouds from depth maps and camera poses.
    """
    
    def __init__(self, camera_matrix: np.ndarray):
        self.K = camera_matrix
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
    
    def depth_to_pointcloud(self, 
                            depth: np.ndarray, 
                            color: np.ndarray,
                            pose: np.ndarray = np.eye(4),
                            depth_scale: float = 1.0,
                            max_depth: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to colored 3D point cloud.
        """
        h, w = depth.shape
        
        # Create mesh grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Normalize depth
        z = depth * depth_scale
        
        # Back-project to 3D
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # Stack points
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
        
        # Filter invalid points
        valid = (z.flatten() > 0) & (z.flatten() < max_depth)
        points = points[valid]
        colors = colors[valid]
        
        # Transform to world coordinates
        points_h = np.hstack([points, np.ones((len(points), 1))])
        points_world = (pose @ points_h.T).T[:, :3]
        
        return points_world, colors


# ============================================================================
# MAIN RECONSTRUCTION SYSTEM
# ============================================================================

class MonocularReconstructor:
    """
    Main reconstruction system combining all components.
    """
    
    def __init__(self, 
                 camera_matrix: Optional[np.ndarray] = None,
                 image_size: Tuple[int, int] = (640, 480),
                 depth_model: str = "depth_anything"):
        
        self.image_size = image_size
        
        # Default camera matrix if not provided
        if camera_matrix is None:
            fx = fy = image_size[0] * 0.8  # Approximate focal length
            cx, cy = image_size[0] / 2, image_size[1] / 2
            self.K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        else:
            self.K = camera_matrix
        
        # Initialize components
        print("Initializing depth estimator...")
        self.depth_estimator = DepthEstimator(model_type=depth_model)
        
        print("Initializing visual odometry...")
        self.vo = VisualOdometry(self.K)
        
        print("Initializing point cloud generator...")
        self.pc_generator = PointCloudGenerator(self.K)
        
        self.plane_detector = PlaneDetector()
        
        # Storage
        self.all_points = []
        self.all_colors = []
        self.frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame and return reconstruction data.
        """
        # Store original dimensions
        original_h, original_w = frame.shape[:2]
        
        # Resize if needed for processing
        if (original_w, original_h) != self.image_size:
            frame_resized = cv2.resize(frame, self.image_size)
        else:
            frame_resized = frame
        
        # Estimate depth (handles textureless surfaces!)
        # Pass original frame for best quality, then resize result
        depth = self.depth_estimator.estimate_depth(frame)
        
        # Resize depth to match processing size for pose estimation
        depth_resized = cv2.resize(depth, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        # Estimate pose using resized frame
        pose = self.vo.process_frame(frame_resized)
        
        # Generate point cloud using resized versions for consistency
        points, colors = self.pc_generator.depth_to_pointcloud(
            depth_resized, frame_resized, pose
        )
        
        # Detect planes (helps with textureless surfaces)
        planes = self.plane_detector.detect_planes(depth_resized)
        
        # Accumulate (subsample for efficiency)
        subsample = max(1, len(points) // 10000)
        self.all_points.append(points[::subsample])
        self.all_colors.append(colors[::subsample])
        
        self.frame_count += 1
        
        return {
            'frame': frame_resized,
            'depth': depth_resized,
            'pose': pose,
            'points': points,
            'colors': colors,
            'planes': planes,
            'frame_count': self.frame_count
        }
    
    def process_images_folder(self, folder_path: str, 
                              extensions: List[str] = ['.jpg', '.png', '.jpeg']):
        """
        Process all images in a folder.
        """
        folder = Path(folder_path)
        image_files = sorted([
            f for f in folder.iterdir() 
            if f.suffix.lower() in extensions
        ])
        
        print(f"Found {len(image_files)} images")
        
        for i, img_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {img_path.name}")
            
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            result = self.process_frame(frame)
            
            # Display progress - resize depth to match frame dimensions
            depth_vis = cv2.normalize(result['depth'], None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            
            # Resize depth visualization to match the processed frame size
            frame_h, frame_w = result['frame'].shape[:2]
            depth_vis = cv2.resize(depth_vis, (frame_w, frame_h))
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
            
            combined = np.hstack([result['frame'], depth_colored])
            cv2.imshow('Reconstruction Progress', combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return self.get_full_pointcloud()
    
    def process_camera(self, camera_id: int = 0, 
                       max_frames: Optional[int] = None):
        """
        Process live camera feed (webcam or USB camera).
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size[1])
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        print("Press 'q' to stop, 's' to save point cloud")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.process_frame(frame)
            
            # Visualize
            depth_vis = cv2.normalize(result['depth'], None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
            
            # Add info overlay
            info_text = f"Frame: {frame_idx} | Points: {len(result['points'])}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            combined = np.hstack([frame, depth_colored])
            cv2.imshow('Live Reconstruction', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_pointcloud('reconstruction.ply')
                print("Saved point cloud!")
            
            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return self.get_full_pointcloud()
    
    def get_full_pointcloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get accumulated point cloud."""
        if not self.all_points:
            return np.array([]), np.array([])
        
        points = np.vstack(self.all_points)
        colors = np.vstack(self.all_colors)
        return points, colors
    
    def save_pointcloud(self, filename: str):
        """Save point cloud to PLY file."""
        points, colors = self.get_full_pointcloud()
        
        if len(points) == 0:
            print("No points to save!")
            return
        
        # Use Open3D if available
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(filename, pcd)
        except ImportError:
            # Manual PLY export
            with open(filename, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                
                for p, c in zip(points, colors):
                    r, g, b = (c * 255).astype(int)
                    f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b}\n")
        
        print(f"Saved {len(points)} points to {filename}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monocular 3D Reconstruction')
    parser.add_argument('--mode', choices=['folder', 'camera'], default='camera',
                       help='Input mode: folder of images or live camera')
    parser.add_argument('--input', type=str, default='./images',
                       help='Input folder path (for folder mode)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (for camera mode)')
    parser.add_argument('--output', type=str, default='reconstruction.ply',
                       help='Output point cloud file')
    parser.add_argument('--depth-model', choices=['depth_anything', 'midas'],
                       default='depth_anything',
                       help='Depth estimation model')
    
    args = parser.parse_args()
    
    # Initialize reconstructor
    reconstructor = MonocularReconstructor(
        depth_model=args.depth_model
    )
    
    # Run reconstruction
    if args.mode == 'folder':
        reconstructor.process_images_folder(args.input)
    else:
        reconstructor.process_camera(args.camera)
    
    # Save results
    reconstructor.save_pointcloud(args.output)
    print(f"\nReconstruction complete! Output: {args.output}")