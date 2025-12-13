"""
Depth-Enhanced 3D Reconstruction for Textureless Surfaces

This system combines:
1. Learning-based depth estimation (Depth Anything V2) for textureless regions
2. Hybrid feature detection (points + lines + edges)
3. Depth-guided feature matching
4. Dense point cloud generation with proper scaling
5. Incremental reconstruction with depth priors

Author: Enhanced pipeline for Kamalnath's textureless surface reconstruction
"""

import numpy as np
import cv2
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - depth estimation will be limited")

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    print("Open3D not available - using basic PLY export")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available - visualization limited")

try:
    import gtsam
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    print("GTSAM not available - bundle adjustment disabled")


# =============================================================================
# CAMERA INTRINSICS
# =============================================================================

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int, height: int) -> 'CameraIntrinsics':
        return cls(
            fx=K[0, 0], fy=K[1, 1],
            cx=K[0, 2], cy=K[1, 2],
            width=width, height=height
        )


# =============================================================================
# DEPTH ESTIMATION MODULE
# =============================================================================

class DepthEstimator:
    """
    Learning-based monocular depth estimation using Depth Anything V2.
    Handles textureless surfaces by learning geometric priors.
    """
    
    def __init__(self, model_type: str = "depth_anything_v2", device: str = "auto"):
        self.model_type = model_type
        self.model = None
        self.processor = None
        
        if device == "auto":
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        if TORCH_AVAILABLE:
            self._load_model()
        else:
            print("Warning: PyTorch not available, depth estimation disabled")
    
    def _load_model(self):
        """Load depth estimation model from HuggingFace"""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            # Use Depth Anything V2 Large for best quality
            model_name = "depth-anything/Depth-Anything-V2-Large-hf"
            
            print(f"Loading {model_name}...")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self.model.to(self.device).eval()
            print(f"Depth model loaded on {self.device}")
            
        except Exception as e:
            print(f"Failed to load Depth Anything V2: {e}")
            print("Trying fallback model...")
            try:
                model_name = "LiheYoung/depth-anything-large-hf"
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
                self.model.to(self.device).eval()
                print(f"Fallback model loaded on {self.device}")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                self.model = None
    
    def estimate(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth from BGR image.
        Returns relative depth map (float32, higher = farther).
        """
        if self.model is None:
            return None
        
        from PIL import Image as PILImage
        
        original_h, original_w = image.shape[:2]
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
        
        # Process
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_h, original_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
        
        return depth.astype(np.float32)
    
    def estimate_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Estimate depth for multiple images"""
        return [self.estimate(img) for img in images]


# =============================================================================
# HYBRID FEATURE DETECTION (Points + Lines + Edges)
# =============================================================================

class HybridFeatureDetector:
    """
    Combines multiple feature types for robust detection on textureless surfaces:
    - Point features (SIFT, ORB) for textured regions
    - Line segments (LSD) for edges on textureless surfaces
    - Edge-based features for boundaries
    """
    
    def __init__(self, use_sift: bool = True, use_orb: bool = True, 
                 use_lines: bool = True, use_edges: bool = True):
        self.use_sift = use_sift
        self.use_orb = use_orb
        self.use_lines = use_lines
        self.use_edges = use_edges
        
        # Point detectors
        if use_sift:
            self.sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02)
        if use_orb:
            self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        
        # Line segment detector
        if use_lines:
            self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        
        # Matchers
        self.bf_l2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # FLANN matcher for SIFT
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect all feature types in an image.
        Returns dictionary with keypoints, descriptors, lines, edges.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        result = {'gray': gray, 'image': image}
        
        # SIFT features
        if self.use_sift:
            kp_sift, desc_sift = self.sift.detectAndCompute(gray, None)
            result['sift_kp'] = kp_sift
            result['sift_desc'] = desc_sift
            print(f"  SIFT: {len(kp_sift)} keypoints")
        
        # ORB features
        if self.use_orb:
            kp_orb, desc_orb = self.orb.detectAndCompute(gray, None)
            result['orb_kp'] = kp_orb
            result['orb_desc'] = desc_orb
            print(f"  ORB: {len(kp_orb)} keypoints")
        
        # Line segments
        if self.use_lines:
            lines, widths, precs, nfas = self.lsd.detect(gray)
            result['lines'] = lines
            if lines is not None:
                print(f"  Lines: {len(lines)} segments")
            else:
                print(f"  Lines: 0 segments")
        
        # Edge map
        if self.use_edges:
            edges = cv2.Canny(gray, 50, 150)
            result['edges'] = edges
            edge_points = np.argwhere(edges > 0)
            print(f"  Edge points: {len(edge_points)}")
        
        return result
    
    def match_features(self, feat1: Dict, feat2: Dict, 
                       ratio_thresh: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match features between two images using multiple methods.
        Returns matched point pairs (pts1, pts2).
        """
        all_pts1, all_pts2 = [], []
        
        # Match SIFT features
        if 'sift_desc' in feat1 and 'sift_desc' in feat2:
            if feat1['sift_desc'] is not None and feat2['sift_desc'] is not None:
                if len(feat1['sift_desc']) > 2 and len(feat2['sift_desc']) > 2:
                    matches = self.flann.knnMatch(
                        feat1['sift_desc'].astype(np.float32),
                        feat2['sift_desc'].astype(np.float32),
                        k=2
                    )
                    for m, n in matches:
                        if m.distance < ratio_thresh * n.distance:
                            pt1 = feat1['sift_kp'][m.queryIdx].pt
                            pt2 = feat2['sift_kp'][m.trainIdx].pt
                            all_pts1.append(pt1)
                            all_pts2.append(pt2)
                    print(f"  SIFT matches: {len(all_pts1)}")
        
        # Match ORB features
        if 'orb_desc' in feat1 and 'orb_desc' in feat2:
            if feat1['orb_desc'] is not None and feat2['orb_desc'] is not None:
                if len(feat1['orb_desc']) > 2 and len(feat2['orb_desc']) > 2:
                    matches = self.bf_hamming.knnMatch(
                        feat1['orb_desc'],
                        feat2['orb_desc'],
                        k=2
                    )
                    orb_count = 0
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < ratio_thresh * n.distance:
                                pt1 = feat1['orb_kp'][m.queryIdx].pt
                                pt2 = feat2['orb_kp'][m.trainIdx].pt
                                all_pts1.append(pt1)
                                all_pts2.append(pt2)
                                orb_count += 1
                    print(f"  ORB matches: {orb_count}")
        
        # Match line segment endpoints
        if 'lines' in feat1 and 'lines' in feat2:
            if feat1['lines'] is not None and feat2['lines'] is not None:
                line_pts1, line_pts2 = self._match_line_endpoints(
                    feat1['lines'], feat2['lines'],
                    feat1['gray'], feat2['gray']
                )
                all_pts1.extend(line_pts1)
                all_pts2.extend(line_pts2)
                print(f"  Line endpoint matches: {len(line_pts1)}")
        
        if len(all_pts1) == 0:
            return np.array([]), np.array([])
        
        pts1 = np.array(all_pts1, dtype=np.float32)
        pts2 = np.array(all_pts2, dtype=np.float32)
        
        # Remove duplicates
        pts1, pts2 = self._remove_duplicate_matches(pts1, pts2)
        
        print(f"  Total unique matches: {len(pts1)}")
        return pts1, pts2
    
    def _match_line_endpoints(self, lines1, lines2, gray1, gray2, 
                              search_radius: float = 30.0) -> Tuple[List, List]:
        """Match line segment endpoints between images"""
        pts1, pts2 = [], []
        
        # Extract endpoints from lines
        endpoints1 = []
        for line in lines1:
            x1, y1, x2, y2 = line[0]
            endpoints1.append((x1, y1))
            endpoints1.append((x2, y2))
        
        endpoints2 = []
        for line in lines2:
            x1, y1, x2, y2 = line[0]
            endpoints2.append((x1, y1))
            endpoints2.append((x2, y2))
        
        if len(endpoints1) == 0 or len(endpoints2) == 0:
            return pts1, pts2
        
        endpoints1 = np.array(endpoints1)
        endpoints2 = np.array(endpoints2)
        
        # Simple nearest neighbor matching with descriptor verification
        for ep1 in endpoints1[:200]:  # Limit to avoid too many matches
            # Find nearby endpoints in image 2
            distances = np.linalg.norm(endpoints2 - ep1, axis=1)
            nearby_idx = np.where(distances < search_radius)[0]
            
            if len(nearby_idx) > 0:
                # Use patch correlation to find best match
                best_idx = nearby_idx[np.argmin(distances[nearby_idx])]
                ep2 = endpoints2[best_idx]
                
                # Verify with patch correlation
                if self._verify_patch_match(gray1, gray2, ep1, ep2):
                    pts1.append(ep1)
                    pts2.append(ep2)
        
        return pts1, pts2
    
    def _verify_patch_match(self, gray1, gray2, pt1, pt2, patch_size: int = 11) -> bool:
        """Verify match using normalized cross-correlation"""
        h1, w1 = gray1.shape
        h2, w2 = gray2.shape
        half = patch_size // 2
        
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        
        # Check bounds
        if (x1 - half < 0 or x1 + half >= w1 or y1 - half < 0 or y1 + half >= h1 or
            x2 - half < 0 or x2 + half >= w2 or y2 - half < 0 or y2 + half >= h2):
            return False
        
        patch1 = gray1[y1-half:y1+half+1, x1-half:x1+half+1].astype(np.float32)
        patch2 = gray2[y2-half:y2+half+1, x2-half:x2+half+1].astype(np.float32)
        
        # Normalized cross-correlation
        patch1 = (patch1 - patch1.mean()) / (patch1.std() + 1e-6)
        patch2 = (patch2 - patch2.mean()) / (patch2.std() + 1e-6)
        
        ncc = np.mean(patch1 * patch2)
        return ncc > 0.7
    
    def _remove_duplicate_matches(self, pts1, pts2, threshold: float = 2.0):
        """Remove duplicate matches"""
        if len(pts1) == 0:
            return pts1, pts2
        
        unique_mask = np.ones(len(pts1), dtype=bool)
        
        for i in range(len(pts1)):
            if not unique_mask[i]:
                continue
            for j in range(i + 1, len(pts1)):
                if not unique_mask[j]:
                    continue
                dist1 = np.linalg.norm(pts1[i] - pts1[j])
                dist2 = np.linalg.norm(pts2[i] - pts2[j])
                if dist1 < threshold and dist2 < threshold:
                    unique_mask[j] = False
        
        return pts1[unique_mask], pts2[unique_mask]


# =============================================================================
# DEPTH-GUIDED FEATURE MATCHING
# =============================================================================

class DepthGuidedMatcher:
    """
    Enhances feature matching using depth information.
    Helps with textureless surfaces by using depth consistency.
    """
    
    def __init__(self, depth_consistency_thresh: float = 0.15):
        self.depth_thresh = depth_consistency_thresh
    
    def filter_matches_by_depth(self, pts1: np.ndarray, pts2: np.ndarray,
                                 depth1: np.ndarray, depth2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter matches using depth consistency.
        Points with similar relative depth should match.
        """
        if depth1 is None or depth2 is None:
            return pts1, pts2
        
        if len(pts1) < 4:
            return pts1, pts2
        
        h1, w1 = depth1.shape
        h2, w2 = depth2.shape
        
        # Get depth values at matched points
        depths1 = []
        depths2 = []
        valid_indices = []
        
        for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            
            if (0 <= x1 < w1 and 0 <= y1 < h1 and
                0 <= x2 < w2 and 0 <= y2 < h2):
                d1 = depth1[y1, x1]
                d2 = depth2[y2, x2]
                if d1 > 0 and d2 > 0:
                    depths1.append(d1)
                    depths2.append(d2)
                    valid_indices.append(i)
        
        if len(valid_indices) < 4:
            return pts1, pts2
        
        depths1 = np.array(depths1)
        depths2 = np.array(depths2)
        
        # Normalize depths
        depths1_norm = (depths1 - depths1.min()) / (depths1.max() - depths1.min() + 1e-6)
        depths2_norm = (depths2 - depths2.min()) / (depths2.max() - depths2.min() + 1e-6)
        
        # Compute depth consistency
        depth_diff = np.abs(depths1_norm - depths2_norm)
        
        # Keep matches with consistent depth
        consistent_mask = depth_diff < self.depth_thresh
        
        # Also keep matches where both points have similar rank order
        rank1 = np.argsort(np.argsort(depths1))
        rank2 = np.argsort(np.argsort(depths2))
        rank_diff = np.abs(rank1 - rank2) / len(rank1)
        rank_consistent = rank_diff < 0.3
        
        final_mask = consistent_mask | rank_consistent
        
        valid_indices = np.array(valid_indices)[final_mask]
        
        print(f"  Depth filtering: {len(pts1)} -> {len(valid_indices)} matches")
        
        return pts1[valid_indices], pts2[valid_indices]
    
    def generate_dense_correspondences(self, depth1: np.ndarray, depth2: np.ndarray,
                                        R: np.ndarray, t: np.ndarray, K: np.ndarray,
                                        grid_step: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dense correspondences using depth maps and known pose.
        Useful for textureless regions.
        """
        h, w = depth1.shape
        
        # Create grid of points
        y_coords, x_coords = np.mgrid[0:h:grid_step, 0:w:grid_step]
        
        pts1 = []
        pts2 = []
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        for y, x in zip(y_coords.ravel(), x_coords.ravel()):
            d = depth1[y, x]
            if d <= 0:
                continue
            
            # Back-project to 3D
            X = (x - cx) * d / fx
            Y = (y - cy) * d / fy
            Z = d
            
            # Transform to camera 2
            P_cam1 = np.array([X, Y, Z])
            P_cam2 = R @ P_cam1 + t.ravel()
            
            if P_cam2[2] <= 0:
                continue
            
            # Project to image 2
            x2 = fx * P_cam2[0] / P_cam2[2] + cx
            y2 = fy * P_cam2[1] / P_cam2[2] + cy
            
            if 0 <= x2 < w and 0 <= y2 < h:
                pts1.append([x, y])
                pts2.append([x2, y2])
        
        return np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32)


# =============================================================================
# DENSE POINT CLOUD GENERATOR
# =============================================================================

class DensePointCloudGenerator:
    """
    Generate dense point clouds from depth maps.
    Essential for textureless surface reconstruction.
    """
    
    def __init__(self, intrinsics: CameraIntrinsics):
        self.K = intrinsics
        self._precompute_projection_maps()
    
    def _precompute_projection_maps(self):
        """Precompute pixel-to-ray mappings"""
        h, w = self.K.height, self.K.width
        
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        self.x_factor = (u - self.K.cx) / self.K.fx
        self.y_factor = (v - self.K.cy) / self.K.fy
    
    def depth_to_pointcloud(self, depth: np.ndarray, color: np.ndarray,
                            pose: Tuple[np.ndarray, np.ndarray] = None,
                            min_depth: float = 0.1, max_depth: float = 100.0,
                            subsample: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to colored 3D point cloud.
        
        Args:
            depth: HxW depth map
            color: HxWx3 BGR image
            pose: (R, t) camera pose in world frame
            min_depth, max_depth: Depth range filter
            subsample: Subsample factor (1 = full resolution)
        
        Returns:
            points: Nx3 world coordinates
            colors: Nx3 RGB colors (0-255)
        """
        h, w = depth.shape
        
        # Ensure maps match depth size
        if self.x_factor.shape != (h, w):
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            x_factor = (u - self.K.cx) / self.K.fx
            y_factor = (v - self.K.cy) / self.K.fy
        else:
            x_factor = self.x_factor
            y_factor = self.y_factor
        
        # Subsample if requested
        if subsample > 1:
            depth = depth[::subsample, ::subsample]
            color = color[::subsample, ::subsample]
            x_factor = x_factor[::subsample, ::subsample]
            y_factor = y_factor[::subsample, ::subsample]
        
        # Create valid mask
        valid = (depth > min_depth) & (depth < max_depth) & np.isfinite(depth)
        
        # Compute 3D coordinates in camera frame
        z = depth[valid]
        x = x_factor[valid] * z
        y = y_factor[valid] * z
        
        points_cam = np.stack([x, y, z], axis=-1)
        
        # Transform to world frame if pose provided
        if pose is not None:
            R, t = pose
            # Camera center in world: C = -R^T @ t
            # Point in world: P_world = R^T @ (P_cam - t) = R^T @ P_cam + C
            points_world = (R.T @ points_cam.T).T - (R.T @ t).T
        else:
            points_world = points_cam
        
        # Extract colors (BGR to RGB)
        colors_bgr = color[valid]
        colors_rgb = colors_bgr[:, ::-1]  # BGR to RGB
        
        return points_world.astype(np.float32), colors_rgb.astype(np.uint8)
    
    def merge_pointclouds(self, pointclouds: List[Tuple[np.ndarray, np.ndarray]],
                          voxel_size: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge multiple point clouds with voxel downsampling.
        """
        all_points = []
        all_colors = []
        
        for points, colors in pointclouds:
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)
        
        if len(all_points) == 0:
            return np.array([]), np.array([])
        
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        
        # Voxel downsampling
        if voxel_size > 0 and O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            
            pcd_down = pcd.voxel_down_sample(voxel_size)
            
            points = np.asarray(pcd_down.points)
            colors = (np.asarray(pcd_down.colors) * 255).astype(np.uint8)
        
        return points, colors


# =============================================================================
# DEPTH SCALE ESTIMATION
# =============================================================================

class DepthScaleEstimator:
    """
    Estimate scale factor to convert relative depth to metric depth.
    Uses sparse reconstruction to calibrate dense depth maps.
    """
    
    @staticmethod
    def estimate_scale(sparse_points: np.ndarray, sparse_pts2d: np.ndarray,
                       depth_map: np.ndarray, K: np.ndarray) -> float:
        """
        Estimate scale factor between relative depth and metric depth.
        
        Args:
            sparse_points: Nx3 triangulated 3D points
            sparse_pts2d: Nx2 corresponding 2D points
            depth_map: Relative depth map from neural network
            K: Camera intrinsic matrix
        
        Returns:
            scale: Scale factor (multiply relative depth by this)
        """
        if len(sparse_points) < 5:
            return 1.0
        
        h, w = depth_map.shape
        
        scales = []
        for pt3d, pt2d in zip(sparse_points, sparse_pts2d):
            x, y = int(pt2d[0]), int(pt2d[1])
            
            if 0 <= x < w and 0 <= y < h:
                relative_depth = depth_map[y, x]
                metric_depth = pt3d[2]  # Z coordinate
                
                if relative_depth > 0 and metric_depth > 0:
                    scale = metric_depth / relative_depth
                    scales.append(scale)
        
        if len(scales) < 3:
            return 1.0
        
        # Use median for robustness
        scale = np.median(scales)
        
        print(f"  Depth scale: {scale:.4f} (from {len(scales)} points)")
        return scale


# =============================================================================
# GEOMETRY UTILITIES (from your original code)
# =============================================================================

def normalize_points(pts):
    """Hartley normalization for numerical stability"""
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    distances = np.sqrt(np.sum(centered**2, axis=1))
    avg_dist = np.mean(distances)
    
    if avg_dist < np.finfo(float).eps:
        scale = 1.0
    else:
        scale = np.sqrt(2.0) / avg_dist
    
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_hom.T).T[:, :2]
    
    return pts_norm, T


def compute_sampson_error(pt1, pt2, F):
    """Compute Sampson distance for fundamental matrix"""
    pt1_h = np.array([pt1[0], pt1[1], 1.0])
    pt2_h = np.array([pt2[0], pt2[1], 1.0])
    
    Fx1 = F @ pt1_h
    Ftx2 = F.T @ pt2_h
    x2tFx1 = pt2_h @ F @ pt1_h
    
    denom = Fx1[0]**2 + Fx1[1]**2 + Ftx2[0]**2 + Ftx2[1]**2
    
    if denom < np.finfo(float).eps:
        return float('inf')
    
    return (x2tFx1**2) / denom


def compute_fundamental_matrix_8point(pts1, pts2):
    """8-point algorithm with normalization"""
    count = pts1.shape[0]
    
    _, T1 = normalize_points(pts1)
    _, T2 = normalize_points(pts2)
    
    ones = np.ones((count, 1))
    pts1_h = np.hstack([pts1, ones])
    pts2_h = np.hstack([pts2, ones])
    pts1_norm = (T1 @ pts1_h.T).T
    pts2_norm = (T2 @ pts2_h.T).T
    
    A = np.zeros((count, 9), dtype=np.float64)
    for i in range(count):
        x1, y1, w1 = pts1_norm[i]
        x2, y2, w2 = pts2_norm[i]
        A[i] = [x1*x2, y1*x2, w1*x2, x1*y2, y1*y2, w1*y2, x1*w2, y1*w2, w1*w2]
    
    _, _, Vt = np.linalg.svd(A)
    F0 = Vt[-1].reshape(3, 3)
    
    # Enforce rank-2
    U, S, Vt = np.linalg.svd(F0)
    S[2] = 0
    F0 = U @ np.diag(S) @ Vt
    
    # Denormalize
    F = T2.T @ F0 @ T1
    
    if abs(F[2, 2]) > np.finfo(float).eps:
        F = F / F[2, 2]
    
    return F


def compute_fundamental_ransac(pts1, pts2, max_iters=2000, threshold=3.0):
    """RANSAC for fundamental matrix"""
    n = pts1.shape[0]
    assert n >= 8
    
    best_F = None
    best_inliers = []
    
    for _ in range(max_iters):
        indices = np.random.choice(n, 8, replace=False)
        
        try:
            F = compute_fundamental_matrix_8point(pts1[indices], pts2[indices])
            
            inliers = []
            for i in range(n):
                error = compute_sampson_error(pts1[i], pts2[i], F)
                if error < threshold:
                    inliers.append(i)
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_F = F
        except:
            continue
    
    # Refine with all inliers
    if len(best_inliers) >= 8:
        try:
            best_F = compute_fundamental_matrix_8point(
                pts1[best_inliers], pts2[best_inliers]
            )
        except:
            pass
    
    mask = np.zeros(n, dtype=bool)
    if best_inliers:
        mask[best_inliers] = True
    
    return best_F, mask


def triangulate_points(P1, P2, pts1, pts2):
    """DLT triangulation"""
    n = pts1.shape[0]
    points_3d = np.zeros((n, 3))
    
    for i in range(n):
        A = np.zeros((4, 4))
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        
        A[0] = x1 * P1[2] - P1[0]
        A[1] = y1 * P1[2] - P1[1]
        A[2] = x2 * P2[2] - P2[0]
        A[3] = y2 * P2[2] - P2[1]
        
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        points_3d[i] = X[:3] / X[3]
    
    return points_3d


def recover_pose(E, pts1, pts2, K):
    """Recover R, t from essential matrix"""
    U, _, Vt = np.linalg.svd(E)
    
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt
    
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2].reshape(3, 1)
    
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    
    solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    
    best_solution = None
    max_good = 0
    best_mask = None
    
    for R, t_test in solutions:
        P2 = K @ np.hstack([R, t_test])
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T
        
        depths1 = points_3d[:, 2]
        points_cam2 = (R @ points_3d.T + t_test).T
        depths2 = points_cam2[:, 2]
        
        good_mask = (depths1 > 0) & (depths2 > 0)
        n_good = np.sum(good_mask)
        
        if n_good > max_good:
            max_good = n_good
            best_solution = (R, t_test)
            best_mask = good_mask
    
    return best_solution[0], best_solution[1], best_mask


# =============================================================================
# MAIN RECONSTRUCTION CLASS
# =============================================================================

class DepthEnhancedReconstruction:
    """
    Main reconstruction class combining all components.
    Handles both textured and textureless surfaces.
    """
    
    def __init__(self, K: np.ndarray, image_size: Tuple[int, int] = None,
                 use_depth: bool = True, use_hybrid_features: bool = True):
        """
        Args:
            K: 3x3 camera intrinsic matrix
            image_size: (width, height) tuple
            use_depth: Use depth estimation for textureless surfaces
            use_hybrid_features: Use hybrid feature detection (points + lines)
        """
        self.K = K
        self.image_size = image_size
        
        # Set up intrinsics
        if image_size:
            self.intrinsics = CameraIntrinsics.from_matrix(K, image_size[0], image_size[1])
        
        # Initialize components
        self.use_depth = use_depth
        if use_depth and TORCH_AVAILABLE:
            print("Initializing depth estimator...")
            self.depth_estimator = DepthEstimator()
        else:
            self.depth_estimator = None
        
        self.use_hybrid = use_hybrid_features
        if use_hybrid_features:
            print("Initializing hybrid feature detector...")
            self.feature_detector = HybridFeatureDetector()
        
        self.depth_matcher = DepthGuidedMatcher()
        
        # Storage
        self.images = []
        self.image_names = []
        self.depths = []
        self.features = []
        self.camera_poses = []  # List of (R, t)
        self.points_3d = []
        self.point_colors = []
        
        print("Reconstruction system initialized")
    
    def load_images(self, folder_path: str, extensions: List[str] = ['.png', '.jpg', '.jpeg']):
        """Load images from folder"""
        folder = Path(folder_path)
        image_files = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in extensions
        ])
        
        print(f"Found {len(image_files)} images")
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                self.images.append(img)
                self.image_names.append(img_path.name)
                print(f"  Loaded: {img_path.name} - Shape: {img.shape}")
        
        # Update image size
        if self.images and self.image_size is None:
            h, w = self.images[0].shape[:2]
            self.image_size = (w, h)
            self.intrinsics = CameraIntrinsics.from_matrix(self.K, w, h)
        
        return len(self.images)
    
    def estimate_all_depths(self):
        """Estimate depth for all images"""
        if self.depth_estimator is None:
            print("Depth estimator not available")
            return
        
        print("\nEstimating depth maps...")
        for i, img in enumerate(self.images):
            print(f"  Processing {self.image_names[i]}...")
            depth = self.depth_estimator.estimate(img)
            self.depths.append(depth)
        
        print(f"Estimated {len(self.depths)} depth maps")
    
    def detect_all_features(self):
        """Detect features in all images"""
        print("\nDetecting features...")
        for i, img in enumerate(self.images):
            print(f"  Processing {self.image_names[i]}...")
            if self.use_hybrid:
                feat = self.feature_detector.detect(img)
            else:
                # Fall back to SIFT only
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sift = cv2.SIFT_create(nfeatures=3000)
                kp, desc = sift.detectAndCompute(gray, None)
                feat = {'sift_kp': kp, 'sift_desc': desc, 'gray': gray, 'image': img}
            self.features.append(feat)
    
    def match_image_pair(self, idx1: int, idx2: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match features between two images.
        Returns (pts1, pts2, mask) where mask indicates inliers.
        """
        print(f"\nMatching images {idx1} <-> {idx2}")
        
        feat1 = self.features[idx1]
        feat2 = self.features[idx2]
        
        # Feature matching
        if self.use_hybrid:
            pts1, pts2 = self.feature_detector.match_features(feat1, feat2)
        else:
            # SIFT matching
            flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            if feat1['sift_desc'] is not None and feat2['sift_desc'] is not None:
                matches = flann.knnMatch(
                    feat1['sift_desc'].astype(np.float32),
                    feat2['sift_desc'].astype(np.float32),
                    k=2
                )
                pts1, pts2 = [], []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        pts1.append(feat1['sift_kp'][m.queryIdx].pt)
                        pts2.append(feat2['sift_kp'][m.trainIdx].pt)
                pts1 = np.array(pts1, dtype=np.float32)
                pts2 = np.array(pts2, dtype=np.float32)
            else:
                pts1, pts2 = np.array([]), np.array([])
        
        if len(pts1) < 8:
            print(f"  Insufficient matches: {len(pts1)}")
            return np.array([]), np.array([]), np.array([])
        
        # Depth-guided filtering
        if self.depths and len(self.depths) > max(idx1, idx2):
            pts1, pts2 = self.depth_matcher.filter_matches_by_depth(
                pts1, pts2, self.depths[idx1], self.depths[idx2]
            )
        
        if len(pts1) < 8:
            print(f"  Insufficient matches after depth filtering: {len(pts1)}")
            return np.array([]), np.array([]), np.array([])
        
        # Geometric verification with RANSAC
        F, mask = compute_fundamental_ransac(pts1, pts2)
        
        if mask is None:
            mask = np.ones(len(pts1), dtype=bool)
        
        print(f"  Final matches: {np.sum(mask)}/{len(pts1)} inliers")
        
        return pts1, pts2, mask
    
    def reconstruct(self, output_dir: str = "./output"):
        """
        Run full reconstruction pipeline.
        """
        print("\n" + "="*70)
        print("STARTING DEPTH-ENHANCED 3D RECONSTRUCTION")
        print("="*70)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Estimate depths
        if self.use_depth:
            self.estimate_all_depths()
        
        # Step 2: Detect features
        self.detect_all_features()
        
        # Step 3: Initialize with first pair
        print("\n--- Initializing with first pair ---")
        pts1, pts2, mask = self.match_image_pair(0, 1)
        
        if len(pts1) < 8:
            print("Failed to initialize - insufficient matches")
            return None
        
        pts1_inlier = pts1[mask]
        pts2_inlier = pts2[mask]
        
        # Compute essential matrix
        E, _ = cv2.findEssentialMat(pts1_inlier, pts2_inlier, self.K, 
                                     method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # Recover pose
        _, R, t, pose_mask = cv2.recoverPose(E, pts1_inlier, pts2_inlier, self.K)
        pose_mask = pose_mask.ravel().astype(bool)
        
        pts1_final = pts1_inlier[pose_mask]
        pts2_final = pts2_inlier[pose_mask]
        
        # Set up projection matrices
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])
        
        # Triangulate sparse points
        points_3d = triangulate_points(P1, P2, pts1_final, pts2_final)
        
        # Store camera poses
        self.camera_poses = [
            (np.eye(3), np.zeros((3, 1))),
            (R, t)
        ]
        
        # Extract colors
        colors = self._extract_colors(self.images[0], pts1_final)
        
        # Store sparse reconstruction
        self.points_3d = list(points_3d)
        self.point_colors = list(colors)
        
        print(f"Initial reconstruction: {len(self.points_3d)} points")
        
        # Step 4: Add dense points from depth maps
        if self.depths:
            print("\n--- Adding dense points from depth maps ---")
            
            # Estimate scale from sparse points
            scale = DepthScaleEstimator.estimate_scale(
                points_3d, pts1_final, self.depths[0], self.K
            )
            
            # Generate dense point cloud generator
            pc_gen = DensePointCloudGenerator(self.intrinsics)
            
            dense_clouds = []
            
            for i, (R_cam, t_cam) in enumerate(self.camera_poses):
                if i >= len(self.depths) or self.depths[i] is None:
                    continue
                
                # Scale depth
                scaled_depth = self.depths[i] * scale
                
                # Generate dense points
                points, colors = pc_gen.depth_to_pointcloud(
                    scaled_depth, self.images[i],
                    pose=(R_cam, t_cam),
                    subsample=4  # Use every 4th pixel for efficiency
                )
                
                dense_clouds.append((points, colors))
                print(f"  Image {i}: {len(points)} dense points")
            
            # Merge dense clouds
            if dense_clouds:
                dense_points, dense_colors = pc_gen.merge_pointclouds(
                    dense_clouds, voxel_size=0.005
                )
                
                # Combine with sparse
                if len(dense_points) > 0:
                    all_points = np.vstack([np.array(self.points_3d), dense_points])
                    all_colors = np.vstack([np.array(self.point_colors), dense_colors])
                else:
                    all_points = np.array(self.points_3d)
                    all_colors = np.array(self.point_colors)
            else:
                all_points = np.array(self.points_3d)
                all_colors = np.array(self.point_colors)
        else:
            all_points = np.array(self.points_3d)
            all_colors = np.array(self.point_colors)
        
        # Step 5: Incrementally add more views
        print("\n--- Adding incremental views ---")
        reconstructed = {0, 1}
        
        for i in range(2, len(self.images)):
            print(f"\nAdding image {i}...")
            
            # Match with previous image
            pts_curr, pts_prev, mask = self.match_image_pair(i, i-1)
            
            if len(pts_curr) < 8 or np.sum(mask) < 6:
                print(f"  Skipping image {i} - insufficient matches")
                continue
            
            pts_curr = pts_curr[mask]
            pts_prev = pts_prev[mask]
            
            # Find 2D-3D correspondences
            # This is simplified - in practice you'd use the stored correspondences
            
            # For now, use essential matrix decomposition
            E, _ = cv2.findEssentialMat(pts_prev, pts_curr, self.K,
                                         method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R_rel, t_rel, _ = cv2.recoverPose(E, pts_prev, pts_curr, self.K)
            
            # Compose with previous pose
            R_prev, t_prev = self.camera_poses[-1]
            R_new = R_rel @ R_prev
            t_new = R_rel @ t_prev + t_rel
            
            self.camera_poses.append((R_new, t_new))
            reconstructed.add(i)
            
            # Triangulate new points
            P_prev = self.K @ np.hstack([R_prev, t_prev])
            P_curr = self.K @ np.hstack([R_new, t_new])
            
            new_points = triangulate_points(P_prev, P_curr, pts_prev, pts_curr)
            new_colors = self._extract_colors(self.images[i], pts_curr)
            
            # Filter by reprojection error
            valid_mask = self._filter_by_reprojection(
                new_points, pts_curr, P_curr, threshold=8.0
            )
            
            new_points = new_points[valid_mask]
            new_colors = new_colors[valid_mask]
            
            self.points_3d.extend(new_points)
            self.point_colors.extend(new_colors)
            
            print(f"  Added {len(new_points)} points (Total: {len(self.points_3d)})")
            
            # Add dense points from depth
            if self.depths and i < len(self.depths) and self.depths[i] is not None:
                if len(new_points) > 5:
                    scale_i = DepthScaleEstimator.estimate_scale(
                        new_points, pts_curr[valid_mask], self.depths[i], self.K
                    )
                    
                    scaled_depth = self.depths[i] * scale_i
                    
                    points, colors = pc_gen.depth_to_pointcloud(
                        scaled_depth, self.images[i],
                        pose=(R_new, t_new),
                        subsample=4
                    )
                    
                    if len(points) > 0:
                        all_points = np.vstack([all_points, points])
                        all_colors = np.vstack([all_colors, colors])
                        print(f"  Added {len(points)} dense points")
        
        print("\n" + "="*70)
        print("RECONSTRUCTION COMPLETE")
        print("="*70)
        print(f"Total points: {len(all_points)}")
        print(f"Total cameras: {len(self.camera_poses)}")
        
        # Save outputs
        self._save_pointcloud(all_points, all_colors, output_path / "reconstruction.ply")
        
        return all_points, all_colors, self.camera_poses
    
    def _extract_colors(self, img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Extract RGB colors at point locations"""
        colors = []
        h, w = img.shape[:2]
        
        for pt in pts:
            x, y = int(pt[0]), int(pt[1])
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            bgr = img[y, x]
            rgb = [bgr[2], bgr[1], bgr[0]]
            colors.append(rgb)
        
        return np.array(colors, dtype=np.uint8)
    
    def _filter_by_reprojection(self, points_3d: np.ndarray, pts_2d: np.ndarray,
                                 P: np.ndarray, threshold: float = 8.0) -> np.ndarray:
        """Filter points by reprojection error"""
        n = len(points_3d)
        points_hom = np.hstack([points_3d, np.ones((n, 1))]).T
        
        projected = (P @ points_hom).T
        projected = projected[:, :2] / projected[:, 2:3]
        
        errors = np.linalg.norm(pts_2d - projected, axis=1)
        
        # Also filter by depth
        depths = points_3d[:, 2]
        valid = (errors < threshold) & (depths > 0.1) & (depths < 100)
        
        return valid
    
    def _save_pointcloud(self, points: np.ndarray, colors: np.ndarray, filepath: Path):
        """Save point cloud to PLY file"""
        if len(points) == 0:
            print("No points to save")
            return
        
        if O3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            o3d.io.write_point_cloud(str(filepath), pcd)
        else:
            # Manual PLY export
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
                    f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        
        print(f"Saved {len(points)} points to {filepath}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_reconstruction(points: np.ndarray, colors: np.ndarray,
                             camera_poses: List[Tuple[np.ndarray, np.ndarray]],
                             title: str = "3D Reconstruction"):
    """Create interactive 3D visualization using Plotly"""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available for visualization")
        return
    
    fig = go.Figure()
    
    # Add point cloud
    if len(points) > 0:
        # Subsample if too many points
        max_points = 100000
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points_vis = points[indices]
            colors_vis = colors[indices]
        else:
            points_vis = points
            colors_vis = colors
        
        colors_rgb = [f'rgb({int(c[0])},{int(c[1])},{int(c[2])})' for c in colors_vis]
        
        fig.add_trace(go.Scatter3d(
            x=points_vis[:, 0],
            y=points_vis[:, 1],
            z=points_vis[:, 2],
            mode='markers',
            marker=dict(size=1.5, color=colors_rgb, opacity=0.8),
            name='Points',
            hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
        ))
    
    # Add cameras
    for i, (R, t) in enumerate(camera_poses):
        C = -R.T @ t
        C = C.ravel()
        
        # Camera axes
        axis_length = 0.3
        
        for axis_idx, (axis_vec, color) in enumerate([
            ([axis_length, 0, 0], 'red'),
            ([0, axis_length, 0], 'green'),
            ([0, 0, axis_length], 'blue')
        ]):
            axis_world = R.T @ np.array(axis_vec).reshape(3, 1)
            fig.add_trace(go.Scatter3d(
                x=[C[0], C[0] + axis_world[0, 0]],
                y=[C[1], C[1] + axis_world[1, 0]],
                z=[C[2], C[2] + axis_world[2, 0]],
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Camera marker
        fig.add_trace(go.Scatter3d(
            x=[C[0]], y=[C[1]], z=[C[2]],
            mode='markers+text',
            marker=dict(size=6, color='yellow', symbol='diamond'),
            text=[f'Cam {i}'],
            textposition='top center',
            name=f'Camera {i}',
            hovertemplate=f'Camera {i}<br>X: {C[0]:.2f}<br>Y: {C[1]:.2f}<br>Z: {C[2]:.2f}<extra></extra>'
        ))
    
    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=1200,
        height=800
    )
    
    fig.show()
    
    # Print statistics
    print(f"\nVisualization Statistics:")
    print(f"  Total points: {len(points)}")
    print(f"  Cameras: {len(camera_poses)}")
    if len(points) > 0:
        print(f"  Bounds:")
        print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Depth-Enhanced 3D Reconstruction')
    parser.add_argument('--input', type=str, default='./input_folder/buddha_images',
                        help='Input folder with images')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--fx', type=float, default=1719.0, help='Focal length X')
    parser.add_argument('--fy', type=float, default=1719.0, help='Focal length Y')
    parser.add_argument('--cx', type=float, default=540.0, help='Principal point X')
    parser.add_argument('--cy', type=float, default=960.0, help='Principal point Y')
    parser.add_argument('--no-depth', action='store_true', help='Disable depth estimation')
    parser.add_argument('--no-hybrid', action='store_true', help='Disable hybrid features')
    
    args = parser.parse_args()
    
    # Camera intrinsic matrix
    K = np.array([
        [args.fx, 0, args.cx],
        [0, args.fy, args.cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Initialize reconstruction
    reconstructor = DepthEnhancedReconstruction(
        K=K,
        use_depth=not args.no_depth,
        use_hybrid_features=not args.no_hybrid
    )
    
    # Load images
    num_images = reconstructor.load_images(args.input)
    
    if num_images < 2:
        print("Need at least 2 images for reconstruction")
        exit(1)
    
    # Run reconstruction
    result = reconstructor.reconstruct(output_dir=args.output)
    
    if result is not None:
        points, colors, poses = result
        
        # Visualize
        visualize_reconstruction(
            points, colors, poses,
            title="Depth-Enhanced 3D Reconstruction"
        )
    else:
        print("Reconstruction failed")