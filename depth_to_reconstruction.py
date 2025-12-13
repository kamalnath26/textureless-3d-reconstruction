"""
Bridge Script: Use Pre-computed Depth Images for 3D Reconstruction

This script takes the output from your depth_processor.py script and creates
a dense 3D reconstruction. It's designed to work with:
- Raw depth images (.npy files)
- 16-bit depth PNGs
- RGB images

The key innovation is using depth information to:
1. Augment sparse feature matching
2. Generate dense point clouds
3. Scale relative depth to metric depth using sparse triangulation

Usage:
    python depth_to_reconstruction.py \
        --rgb-folder ./input_folder/buddha_images \
        --depth-folder ./output/depth_images \
        --output ./reconstruction_output

Author: For Kamalnath's textureless surface reconstruction project
"""

import numpy as np
import cv2
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import argparse

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class ReconstructionConfig:
    """Configuration for reconstruction"""
    # Camera intrinsics
    fx: float = 1719.0
    fy: float = 1719.0
    cx: float = 540.0
    cy: float = 960.0
    
    # Depth processing
    depth_scale: float = 1.0  # Will be estimated
    min_depth: float = 0.1
    max_depth: float = 50.0
    
    # Feature matching
    match_ratio: float = 0.75
    ransac_threshold: float = 3.0
    
    # Point cloud
    voxel_size: float = 0.005
    subsample_factor: int = 2
    
    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


class DepthImageLoader:
    """Load depth images from various formats"""
    
    @staticmethod
    def load_depth(filepath: Path) -> Optional[np.ndarray]:
        """Load depth from .npy, .png (16-bit), or .exr"""
        if filepath.suffix == '.npy':
            return np.load(str(filepath)).astype(np.float32)
        
        elif filepath.suffix == '.png':
            # Try 16-bit first
            depth = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
            if depth is not None:
                # Assuming millimeters, convert to meters
                return depth.astype(np.float32) / 1000.0
        
        elif filepath.suffix in ['.exr', '.EXR']:
            depth = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
            if depth is not None:
                return depth.astype(np.float32)
        
        return None
    
    @staticmethod
    def find_matching_depth(rgb_name: str, depth_folder: Path) -> Optional[Path]:
        """Find depth file matching an RGB image name"""
        stem = Path(rgb_name).stem
        
        # Try various naming patterns
        patterns = [
            f"{stem}_depth.npy",
            f"{stem}_depth.png", 
            f"{stem}.npy",
            f"{stem}.png",
            f"depth_{stem}.npy",
            f"depth_{stem}.png",
        ]
        
        for pattern in patterns:
            depth_path = depth_folder / pattern
            if depth_path.exists():
                return depth_path
        
        return None


class SparseReconstructor:
    """
    Sparse reconstruction using traditional SfM.
    This provides the scale reference for dense depth integration.
    """
    
    def __init__(self, config: ReconstructionConfig):
        self.config = config
        self.K = config.K
        
        # Feature detector - use more features with lower threshold
        self.sift = cv2.SIFT_create(
            nfeatures=8000, 
            contrastThreshold=0.01,  # Lower threshold for more features
            edgeThreshold=15,
            sigma=1.6
        )
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=100)
        )
    
    def detect_and_match(self, img1: np.ndarray, img2: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, List]:
        """Detect SIFT features and match"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better feature detection on low-contrast images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray1 = clahe.apply(gray1)
        gray2 = clahe.apply(gray2)
        
        kp1, desc1 = self.sift.detectAndCompute(gray1, None)
        kp2, desc2 = self.sift.detectAndCompute(gray2, None)
        
        print(f"  Detected features: {len(kp1)} / {len(kp2)}")
        
        if desc1 is None or desc2 is None or len(desc1) < 10 or len(desc2) < 10:
            return np.array([]), np.array([]), []
        
        matches = self.flann.knnMatch(
            desc1.astype(np.float32),
            desc2.astype(np.float32),
            k=2
        )
        
        # Ratio test with adaptive threshold
        good_matches = []
        pts1, pts2 = [], []
        for match_pair in matches:
            if len(match_pair) != 2:
                continue
            m, n = match_pair
            if m.distance < self.config.match_ratio * n.distance:
                good_matches.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
        
        return np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32), good_matches
    
    def compute_pose(self, pts1: np.ndarray, pts2: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute relative pose between two views"""
        if len(pts1) < 8:
            return None, None, None
        
        # Find essential matrix with more lenient threshold
        E, mask_e = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=2.0  # Increased from 1.0
        )
        
        if E is None:
            return None, None, None
        
        # Check if we have enough inliers
        if mask_e is None or np.sum(mask_e) < 8:
            return None, None, None
        
        # Recover pose
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)
        
        # Combine masks
        mask = mask_e.ravel().astype(bool) & mask_pose.ravel().astype(bool)
        
        # Verify we have enough inliers
        if np.sum(mask) < 8:
            # Fall back to essential matrix mask only
            mask = mask_e.ravel().astype(bool)
        
        return R, t, mask
    
    def triangulate(self, pts1: np.ndarray, pts2: np.ndarray,
                   R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Triangulate 3D points"""
        # Ensure we have valid input
        if len(pts1) == 0 or len(pts2) == 0:
            return np.array([]).reshape(0, 3)
        
        # Ensure points are 2D arrays with shape (N, 2)
        pts1 = np.atleast_2d(pts1).astype(np.float64)
        pts2 = np.atleast_2d(pts2).astype(np.float64)
        
        if pts1.shape[0] < 2 or pts1.shape[1] != 2:
            return np.array([]).reshape(0, 3)
        
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])
        
        # OpenCV expects (2, N) arrays
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        return points_3d
    
    def filter_points(self, points_3d: np.ndarray, pts1: np.ndarray, pts2: np.ndarray,
                     R: np.ndarray, t: np.ndarray, max_error: float = 5.0
                     ) -> np.ndarray:
        """Filter points by reprojection error and depth"""
        n = len(points_3d)
        valid = np.ones(n, dtype=bool)
        
        # Depth in camera 1
        valid &= (points_3d[:, 2] > self.config.min_depth)
        valid &= (points_3d[:, 2] < self.config.max_depth)
        
        # Depth in camera 2
        pts_cam2 = (R @ points_3d.T + t).T
        valid &= (pts_cam2[:, 2] > self.config.min_depth)
        
        # Reprojection error
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])
        
        pts_hom = np.hstack([points_3d, np.ones((n, 1))]).T
        
        proj1 = (P1 @ pts_hom).T
        proj1 = proj1[:, :2] / proj1[:, 2:3]
        err1 = np.linalg.norm(pts1 - proj1, axis=1)
        
        proj2 = (P2 @ pts_hom).T
        proj2 = proj2[:, :2] / proj2[:, 2:3]
        err2 = np.linalg.norm(pts2 - proj2, axis=1)
        
        valid &= (err1 < max_error) & (err2 < max_error)
        
        return valid


class DenseReconstructor:
    """
    Dense reconstruction using depth maps.
    Combines sparse SfM with dense depth estimation.
    """
    
    def __init__(self, config: ReconstructionConfig):
        self.config = config
        self.K = config.K
        
        # Precompute projection factors
        self._projection_cache = {}
    
    def _get_projection_factors(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get cached projection factors for given image size"""
        key = (h, w)
        if key not in self._projection_cache:
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            x_factor = (u - self.config.cx) / self.config.fx
            y_factor = (v - self.config.cy) / self.config.fy
            self._projection_cache[key] = (x_factor, y_factor)
        return self._projection_cache[key]
    
    def estimate_scale(self, sparse_points: np.ndarray, sparse_pts2d: np.ndarray,
                      depth_map: np.ndarray) -> float:
        """
        Estimate scale factor to convert relative/metric depth.
        Uses sparse triangulated points as reference.
        """
        h, w = depth_map.shape
        scales = []
        
        for pt3d, pt2d in zip(sparse_points, sparse_pts2d):
            x, y = int(pt2d[0]), int(pt2d[1])
            
            if 0 <= x < w and 0 <= y < h:
                depth_nn = depth_map[y, x]
                depth_sparse = pt3d[2]  # Z coordinate
                
                if depth_nn > 0 and depth_sparse > 0:
                    scale = depth_sparse / depth_nn
                    if 0.001 < scale < 1000:  # Sanity check
                        scales.append(scale)
        
        if len(scales) < 3:
            print("Warning: Too few scale samples, using default scale=1.0")
            return 1.0
        
        # Use median for robustness
        scale = np.median(scales)
        print(f"Estimated depth scale: {scale:.6f} (from {len(scales)} samples)")
        
        return scale
    
    def depth_to_pointcloud(self, depth: np.ndarray, color: np.ndarray,
                            pose: Tuple[np.ndarray, np.ndarray] = None,
                            scale: float = 1.0,
                            subsample: int = 1
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to colored point cloud.
        
        Args:
            depth: HxW depth map (relative or metric)
            color: HxWx3 BGR image
            pose: (R, t) camera pose
            scale: Scale factor for depth
            subsample: Subsampling factor
        """
        h, w = depth.shape
        
        # Get projection factors
        x_factor, y_factor = self._get_projection_factors(h, w)
        
        # Subsample
        if subsample > 1:
            depth = depth[::subsample, ::subsample]
            color = color[::subsample, ::subsample]
            x_factor = x_factor[::subsample, ::subsample]
            y_factor = y_factor[::subsample, ::subsample]
        
        # Scale depth
        depth_scaled = depth * scale
        
        # Valid mask
        valid = (depth_scaled > self.config.min_depth) & \
                (depth_scaled < self.config.max_depth) & \
                np.isfinite(depth_scaled)
        
        # Back-project to 3D
        z = depth_scaled[valid]
        x = x_factor[valid] * z
        y = y_factor[valid] * z
        
        points_cam = np.stack([x, y, z], axis=-1)
        
        # Transform to world frame
        if pose is not None:
            R, t = pose
            # P_world = R^T @ (P_cam - t) for pose [R|t] convention
            # But we store R, t where camera center C = -R^T @ t
            # So P_world = R^T @ P_cam + C = R^T @ P_cam - R^T @ t
            points_world = (R.T @ points_cam.T).T - (R.T @ t).ravel()
        else:
            points_world = points_cam
        
        # Colors (BGR to RGB)
        colors_bgr = color[valid]
        colors_rgb = colors_bgr[:, ::-1]
        
        return points_world.astype(np.float32), colors_rgb.astype(np.uint8)
    
    def merge_pointclouds(self, clouds: List[Tuple[np.ndarray, np.ndarray]],
                         voxel_size: float = 0.005
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge and downsample multiple point clouds"""
        all_points = []
        all_colors = []
        
        for points, colors in clouds:
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)
        
        if len(all_points) == 0:
            return np.array([]), np.array([])
        
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        
        # Voxel downsampling with Open3D
        if voxel_size > 0 and O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            
            pcd_down = pcd.voxel_down_sample(voxel_size)
            
            # Statistical outlier removal
            pcd_clean, _ = pcd_down.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            
            points = np.asarray(pcd_clean.points)
            colors = (np.asarray(pcd_clean.colors) * 255).astype(np.uint8)
        
        return points, colors


class DepthToReconstructionPipeline:
    """
    Main pipeline that combines sparse SfM with dense depth.
    """
    
    def __init__(self, config: ReconstructionConfig = None):
        self.config = config or ReconstructionConfig()
        self.sparse = SparseReconstructor(self.config)
        self.dense = DenseReconstructor(self.config)
        
        # Data storage
        self.images = []
        self.image_names = []
        self.depths = []
        self.camera_poses = []
        
    def load_data(self, rgb_folder: str, depth_folder: str):
        """Load RGB images and corresponding depth maps"""
        rgb_path = Path(rgb_folder)
        depth_path = Path(depth_folder)
        
        # Get RGB images
        rgb_files = sorted([
            f for f in rgb_path.iterdir()
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ])
        
        print(f"Found {len(rgb_files)} RGB images")
        
        for rgb_file in rgb_files:
            # Load RGB
            img = cv2.imread(str(rgb_file))
            if img is None:
                continue
            
            # Find matching depth
            depth_file = DepthImageLoader.find_matching_depth(rgb_file.name, depth_path)
            
            if depth_file is not None:
                depth = DepthImageLoader.load_depth(depth_file)
                if depth is not None:
                    # Resize depth to match RGB if needed
                    if depth.shape[:2] != img.shape[:2]:
                        depth = cv2.resize(depth, (img.shape[1], img.shape[0]),
                                          interpolation=cv2.INTER_LINEAR)
                    
                    self.images.append(img)
                    self.depths.append(depth)
                    self.image_names.append(rgb_file.name)
                    print(f"  Loaded: {rgb_file.name} with depth")
            else:
                print(f"  Warning: No depth found for {rgb_file.name}")
        
        print(f"Loaded {len(self.images)} image-depth pairs")
        return len(self.images)
    
    def reconstruct(self) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Run full reconstruction pipeline.
        Returns (points, colors, camera_poses).
        """
        if len(self.images) < 2:
            print("Need at least 2 images")
            return None, None, None
        
        print("\n" + "="*70)
        print("DEPTH-ENHANCED RECONSTRUCTION PIPELINE")
        print("="*70)
        
        all_clouds = []
        
        # ===== Step 1: Initialize with first pair =====
        print("\n--- Step 1: Initialize with first pair ---")
        
        img1, img2 = self.images[0], self.images[1]
        depth1, depth2 = self.depths[0], self.depths[1]
        
        # Sparse matching
        pts1, pts2, matches = self.sparse.detect_and_match(img1, img2)
        print(f"Feature matches: {len(pts1)}")
        
        if len(pts1) < 8:
            print("Insufficient matches for initialization")
            return None, None, None
        
        # Compute pose
        R, t, mask = self.sparse.compute_pose(pts1, pts2)
        if R is None:
            print("Pose estimation failed")
            return None, None, None
        
        pts1_inlier = pts1[mask]
        pts2_inlier = pts2[mask]
        print(f"Inliers: {len(pts1_inlier)}")
        
        # Check if we have enough inliers
        if len(pts1_inlier) < 8:
            print(f"Warning: Very few inliers ({len(pts1_inlier)}), reconstruction may be unreliable")
            # Continue anyway - we'll rely more on depth
        
        # Triangulate sparse points
        sparse_points = self.sparse.triangulate(pts1_inlier, pts2_inlier, R, t)
        
        if len(sparse_points) == 0:
            print("Triangulation failed, using depth-only mode")
            # Fall back to depth-only reconstruction
            valid = np.array([])
            pts1_valid = np.array([]).reshape(0, 2)
            pts2_valid = np.array([]).reshape(0, 2)
            sparse_points = np.array([]).reshape(0, 3)
        else:
            # Filter
            valid = self.sparse.filter_points(sparse_points, pts1_inlier, pts2_inlier, R, t)
            sparse_points = sparse_points[valid]
            pts1_valid = pts1_inlier[valid]
            pts2_valid = pts2_inlier[valid]
        
        print(f"Valid sparse points: {len(sparse_points)}")
        
        # Store poses
        self.camera_poses = [
            (np.eye(3), np.zeros((3, 1))),
            (R, t)
        ]
        
        # ===== Step 2: Estimate depth scale =====
        print("\n--- Step 2: Estimate depth scale ---")
        
        if len(sparse_points) >= 3:
            scale1 = self.dense.estimate_scale(sparse_points, pts1_valid, depth1)
            scale2 = self.dense.estimate_scale(sparse_points, pts2_valid, depth2)
            avg_scale = (scale1 + scale2) / 2
        else:
            print("Warning: Not enough sparse points for scale estimation")
            print("Using default scale = 1.0 (depth assumed to be in meters)")
            avg_scale = 1.0
        
        print(f"Average scale: {avg_scale:.6f}")
        
        # ===== Step 3: Generate dense point clouds =====
        print("\n--- Step 3: Generate dense point clouds ---")
        
        # First camera
        points1, colors1 = self.dense.depth_to_pointcloud(
            depth1, img1,
            pose=self.camera_poses[0],
            scale=avg_scale,
            subsample=self.config.subsample_factor
        )
        all_clouds.append((points1, colors1))
        print(f"Camera 0: {len(points1)} points")
        
        # Second camera
        points2, colors2 = self.dense.depth_to_pointcloud(
            depth2, img2,
            pose=self.camera_poses[1],
            scale=avg_scale,
            subsample=self.config.subsample_factor
        )
        all_clouds.append((points2, colors2))
        print(f"Camera 1: {len(points2)} points")
        
        # ===== Step 4: Add remaining views =====
        print("\n--- Step 4: Add remaining views ---")
        
        for i in range(2, len(self.images)):
            print(f"\nProcessing image {i}...")
            
            img_curr = self.images[i]
            depth_curr = self.depths[i]
            img_prev = self.images[i-1]
            
            # Match with previous
            pts_prev, pts_curr, _ = self.sparse.detect_and_match(img_prev, img_curr)
            
            if len(pts_prev) < 8:
                print(f"  Skipping - insufficient matches ({len(pts_prev)})")
                continue
            
            # Compute relative pose
            R_rel, t_rel, mask = self.sparse.compute_pose(pts_prev, pts_curr)
            
            if R_rel is None:
                print(f"  Skipping - pose estimation failed")
                continue
            
            # Apply mask and ensure we have enough points
            pts_prev_in = pts_prev[mask]
            pts_curr_in = pts_curr[mask]
            
            if len(pts_prev_in) < 8:
                print(f"  Skipping - insufficient inliers ({len(pts_prev_in)})")
                continue
            
            # Compose with previous pose
            R_prev, t_prev = self.camera_poses[-1]
            R_curr = R_rel @ R_prev
            t_curr = R_rel @ t_prev + t_rel
            
            self.camera_poses.append((R_curr, t_curr))
            
            # Triangulate for scale estimation
            sparse_pts = self.sparse.triangulate(pts_prev_in, pts_curr_in, R_rel, t_rel)
            
            # Check if triangulation succeeded
            if len(sparse_pts) == 0:
                print(f"  Warning: Triangulation produced no points, using previous scale")
                scale_i = avg_scale
            else:
                # Transform to world coordinates
                sparse_pts_world = (R_prev.T @ sparse_pts.T).T - (R_prev.T @ t_prev).ravel()
                
                # Filter valid points (positive depth, reasonable range)
                valid_depth = (sparse_pts_world[:, 2] > 0.1) & (sparse_pts_world[:, 2] < 100)
                sparse_pts_world = sparse_pts_world[valid_depth]
                pts_curr_for_scale = pts_curr_in[valid_depth]
                
                if len(sparse_pts_world) >= 3:
                    # Estimate scale for this view
                    scale_i = self.dense.estimate_scale(
                        sparse_pts_world, pts_curr_for_scale, depth_curr
                    )
                else:
                    print(f"  Warning: Not enough valid points for scale, using previous")
                    scale_i = avg_scale
            
            # Use running average of scales for stability
            avg_scale = 0.7 * avg_scale + 0.3 * scale_i
            
            # Generate dense cloud
            points_i, colors_i = self.dense.depth_to_pointcloud(
                depth_curr, img_curr,
                pose=self.camera_poses[-1],
                scale=avg_scale,
                subsample=self.config.subsample_factor
            )
            all_clouds.append((points_i, colors_i))
            print(f"  Camera {i}: {len(points_i)} points")
        
        # ===== Step 5: Merge all clouds =====
        print("\n--- Step 5: Merge and clean point cloud ---")
        
        final_points, final_colors = self.dense.merge_pointclouds(
            all_clouds, voxel_size=self.config.voxel_size
        )
        
        print(f"\nFinal reconstruction: {len(final_points)} points, {len(self.camera_poses)} cameras")
        
        return final_points, final_colors, self.camera_poses
    
    def save_reconstruction(self, points: np.ndarray, colors: np.ndarray,
                           output_path: str):
        """Save point cloud to PLY file"""
        if len(points) == 0:
            print("No points to save")
            return
        
        filepath = Path(output_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            o3d.io.write_point_cloud(str(filepath), pcd)
        else:
            with open(filepath, 'w') as f:
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
                    f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
        
        print(f"Saved to {filepath}")


def visualize_with_plotly(points: np.ndarray, colors: np.ndarray,
                         camera_poses: List[Tuple[np.ndarray, np.ndarray]],
                         title: str = "3D Reconstruction"):
    """Interactive 3D visualization"""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available")
        return
    
    fig = go.Figure()
    
    # Subsample for visualization
    max_vis = 200000
    if len(points) > max_vis:
        idx = np.random.choice(len(points), max_vis, replace=False)
        vis_points = points[idx]
        vis_colors = colors[idx]
    else:
        vis_points = points
        vis_colors = colors
    
    # Point cloud
    color_strings = [f'rgb({c[0]},{c[1]},{c[2]})' for c in vis_colors]
    
    fig.add_trace(go.Scatter3d(
        x=vis_points[:, 0],
        y=vis_points[:, 1],
        z=vis_points[:, 2],
        mode='markers',
        marker=dict(size=1, color=color_strings, opacity=0.9),
        name='Points'
    ))
    
    # Cameras
    for i, (R, t) in enumerate(camera_poses):
        C = -R.T @ t
        C = C.ravel()
        
        # Camera frustum
        axis_len = 0.2
        for vec, col in [([axis_len,0,0], 'red'), ([0,axis_len,0], 'green'), ([0,0,axis_len], 'blue')]:
            end = C + (R.T @ np.array(vec))
            fig.add_trace(go.Scatter3d(
                x=[C[0], end[0]], y=[C[1], end[1]], z=[C[2], end[2]],
                mode='lines', line=dict(color=col, width=3),
                showlegend=False
            ))
        
        fig.add_trace(go.Scatter3d(
            x=[C[0]], y=[C[1]], z=[C[2]],
            mode='markers+text',
            marker=dict(size=5, color='yellow'),
            text=[f'{i}'], textposition='top center',
            name=f'Cam {i}'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(aspectmode='data'),
        width=1400, height=900
    )
    
    fig.show()


def main():
    parser = argparse.ArgumentParser(description='Depth to 3D Reconstruction')
    parser.add_argument('--rgb-folder', type=str, required=True,
                        help='Folder with RGB images')
    parser.add_argument('--depth-folder', type=str, required=True,
                        help='Folder with depth images')
    parser.add_argument('--output', type=str, default='./output/reconstruction.ply',
                        help='Output PLY file path')
    parser.add_argument('--fx', type=float, default=1719.0)
    parser.add_argument('--fy', type=float, default=1719.0)
    parser.add_argument('--cx', type=float, default=540.0)
    parser.add_argument('--cy', type=float, default=960.0)
    parser.add_argument('--voxel-size', type=float, default=0.005)
    parser.add_argument('--subsample', type=int, default=2)
    parser.add_argument('--no-vis', action='store_true')
    
    args = parser.parse_args()
    
    # Configuration
    config = ReconstructionConfig(
        fx=args.fx, fy=args.fy,
        cx=args.cx, cy=args.cy,
        voxel_size=args.voxel_size,
        subsample_factor=args.subsample
    )
    
    # Run pipeline
    pipeline = DepthToReconstructionPipeline(config)
    
    num_loaded = pipeline.load_data(args.rgb_folder, args.depth_folder)
    if num_loaded < 2:
        print("Failed to load sufficient data")
        return
    
    points, colors, poses = pipeline.reconstruct()
    
    if points is not None and len(points) > 0:
        # Save
        pipeline.save_reconstruction(points, colors, args.output)
        
        # Visualize
        if not args.no_vis:
            visualize_with_plotly(points, colors, poses, "Depth-Enhanced Reconstruction")
    else:
        print("Reconstruction failed")


if __name__ == "__main__":
    main()