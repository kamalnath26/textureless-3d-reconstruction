import numpy as np
import cv2
import json
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIG ---
OUTPUT_DIR = "loftr_output"
DEPTH_DIR = "depth_resized"
IMAGE_DIR = "images_resized"

# Load data
print("=== Loading Data ===")

# Load matches
matches_data = np.load(f"{OUTPUT_DIR}/loftr_matches.npz")
all_matches = {}
pairs = set()

for key in matches_data.keys():
    if key.endswith('_mkpts0'):
        pair_str = key.replace('_mkpts0', '')
        i, j = map(int, pair_str.split('_')[1:])
        pairs.add((i, j))

for (i, j) in pairs:
    mkpts0 = matches_data[f"pair_{i}_{j}_mkpts0"]
    mkpts1 = matches_data[f"pair_{i}_{j}_mkpts1"]
    conf = matches_data[f"pair_{i}_{j}_conf"]
    all_matches[(i, j)] = (mkpts0, mkpts1, conf)

print(f"Loaded {len(all_matches)} match pairs")

# Load initial poses
initial_poses = np.load(f"{OUTPUT_DIR}/initial_poses.npy")
n_cameras = len(initial_poses)
print(f"Loaded {n_cameras} camera poses")

# Load intrinsics
with open("intrinsics.json", "r") as f:
    intrinsics = json.load(f)

K = np.array([
    [intrinsics['fx'], 0, intrinsics['cx']],
    [0, intrinsics['fy'], intrinsics['cy']],
    [0, 0, 1]
])

print(f"Intrinsics:\n{K}")

# # Load depth maps
# print("\nLoading depth maps...")
# image_files = sorted(Path(IMAGE_DIR).glob("*.png"))
# depth_maps = []

# for img_file in image_files:
#     depth_file = Path(DEPTH_DIR) / img_file.name
#     if not depth_file.exists():
#         print(f"ERROR: Missing depth for {img_file.name}")
#         exit(1)
    
#     depth = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
#     if depth is None:
#         print(f"ERROR: Cannot load depth {depth_file}")
#         exit(1)
    
#     # Normalize to [0, 1]
#     depth = depth.astype(np.float32) / 255.0
#     depth_maps.append(depth)

# Load depth maps
# Load depth maps
print("\nLoading depth maps...")
image_files = sorted(Path(IMAGE_DIR).glob("*.png"))
depth_maps = []

for img_file in image_files:
    depth_file = Path(DEPTH_DIR) / img_file.name
    if not depth_file.exists():
        print(f"ERROR: Missing depth for {img_file.name}")
        exit(1)
    
    # Load RGB depth
    depth_rgb = cv2.imread(str(depth_file))
    if depth_rgb is None:
        print(f"ERROR: Cannot load depth {depth_file}")
        exit(1)
    
    # Decode using all 3 channels (luminance formula)
    b, g, r = depth_rgb[:,:,0], depth_rgb[:,:,1], depth_rgb[:,:,2]
    depth_gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Normalize to [0-1]
    depth_normalized = depth_gray.astype(np.float32) / 255.0
    
    depth_maps.append(depth_normalized)

print(f"Loaded {len(depth_maps)} depth maps")

print(f"Loaded {len(depth_maps)} depth maps")
with open("per_image_depth_ranges.json", "r") as f:
    depth_ranges = json.load(f)

depth_lookup = {item['img_id']: item for item in depth_ranges}

#-------------------Temp_-------------------#
print("\n=== DEBUG: Checking Depth Variation ===")
for i in range(min(3, len(depth_maps))):
    if i in depth_lookup:
        depth_info = depth_lookup[i]
        depth_normalized = depth_maps[i]
        depth_relative = depth_normalized * (depth_info['max'] - depth_info['min']) + depth_info['min']
        
        print(f"\nImage {i}:")
        print(f"  Normalized depth: [{depth_normalized.min():.3f}, {depth_normalized.max():.3f}]")
        print(f"  Std: {depth_normalized.std():.3f}")
        print(f"  Depth range from JSON: [{depth_info['min']:.3f}, {depth_info['max']:.3f}]")
        print(f"  Relative depth: [{depth_relative.min():.3f}, {depth_relative.max():.3f}]")
        print(f"  Median depth: {np.median(depth_relative):.3f}")
#_------------------------------#
# Load per-image depth ranges


# --- HELPER FUNCTIONS ---
def backproject_point(u, v, depth, K):
    """Backproject pixel to 3D point in camera frame"""
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.array([x, y, z])

def sample_depth_value(depth_map, u, v, depth_range):
    """Sample depth at pixel location and convert to relative depth"""
    h, w = depth_map.shape
    u_int = int(np.clip(u, 0, w-1))
    v_int = int(np.clip(v, 0, h-1))
    
    depth_normalized = depth_map[v_int, u_int]
    
    # Map to relative depth using per-image range
    min_d = depth_range['min']
    max_d = depth_range['max']
    depth_relative = depth_normalized * (max_d - min_d) + min_d
    
    return depth_relative

def pose_to_vec(pose):
    """Convert 4x4 pose matrix to 6D vector [rx, ry, rz, tx, ty, tz]"""
    R_mat = pose[:3, :3]
    t = pose[:3, 3]
    rotvec = R.from_matrix(R_mat).as_rotvec()
    return np.concatenate([rotvec, t])

def vec_to_pose(vec):
    """Convert 6D vector to 4x4 pose matrix"""
    rotvec = vec[:3]
    t = vec[3:6]
    R_mat = R.from_rotvec(rotvec).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = R_mat
    pose[:3, 3] = t
    return pose

# --- PREPARE OPTIMIZATION DATA ---
print("\n=== Preparing Optimization Data ===")

# Build list of all constraints
constraints = []

for (i, j), (mkpts0, mkpts1, conf) in all_matches.items():
    # Sample every Nth match to speed up (still plenty of constraints)
    step = max(1, len(mkpts0) // 500)  # Use at most 200 matches per pair
    
    for idx in range(0, len(mkpts0), step):
        u_i, v_i = mkpts0[idx]
        u_j, v_j = mkpts1[idx]
        
        # Get depth values
        if i not in depth_lookup or j not in depth_lookup:
            continue
        
        depth_val_i = sample_depth_value(depth_maps[i], u_i, v_i, depth_lookup[i])
        depth_val_j = sample_depth_value(depth_maps[j], u_j, v_j, depth_lookup[j])
        
        if depth_val_i < 0.01 or depth_val_j < 0.01:
            continue
        
        constraints.append({
            'cam_i': i,
            'cam_j': j,
            'pixel_i': (u_i, v_i),
            'pixel_j': (u_j, v_j),
            'depth_i': depth_val_i,
            'depth_j': depth_val_j,
            'weight': np.sqrt(conf[idx])  # Weight by confidence
        })

print(f"Created {len(constraints)} constraints from {len(all_matches)} pairs")

# --- OPTIMIZATION ---
print("\n=== Setting Up Optimization ===")

def residuals(x, constraints, K):
    """
    Compute residuals for all constraints
    
    x layout:
    - First 6*n_cameras elements: camera poses as [rx,ry,rz,tx,ty,tz] for each camera
    - Last n_cameras elements: depth scales for each camera
    """
    n_cams = n_cameras
    
    # Extract poses
    poses = []
    for i in range(n_cams):
        pose_vec = x[i*6:(i+1)*6]
        poses.append(vec_to_pose(pose_vec))
    
    # Extract depth scales
    depth_scales = x[n_cams*6:]
    
    # Compute residuals
    errors = []
    
    for c in constraints:
        i = c['cam_i']
        j = c['cam_j']
        
        # Backproject from camera i
        u_i, v_i = c['pixel_i']
        depth_i = c['depth_i'] * depth_scales[i]
        P_i_cam = backproject_point(u_i, v_i, depth_i, K)
        P_i_world = poses[i][:3, :3] @ P_i_cam + poses[i][:3, 3]
        
        # Backproject from camera j
        u_j, v_j = c['pixel_j']
        depth_j = c['depth_j'] * depth_scales[j]
        P_j_cam = backproject_point(u_j, v_j, depth_j, K)
        P_j_world = poses[j][:3, :3] @ P_j_cam + poses[j][:3, 3]
        
        # Error: 3D points should match
        error = P_i_world - P_j_world
        
        # Weight by confidence
        error *= c['weight']
        
        errors.extend(error)
    
    return np.array(errors)

# --- INITIAL VALUES ---
print("Setting initial values...")

# Convert poses to vectors
x0 = []
for pose in initial_poses:
    x0.extend(pose_to_vec(pose))

# Initial depth scales (all 1.0)
x0.extend([1.0] * n_cameras)

x0 = np.array(x0)

print(f"Total variables: {len(x0)}")
print(f"  - Pose variables: {n_cameras * 6} (6 per camera)")
print(f"  - Depth scale variables: {n_cameras}")
print(f"Total constraints: {len(constraints)}")
print(f"Total residuals: {len(constraints) * 3} (3 per constraint)")

# --- SET BOUNDS ---
print("\nSetting bounds...")

# Initialize bounds
lower_bounds = [-np.inf] * len(x0)
upper_bounds = [np.inf] * len(x0)

# Fix camera 0 pose (gauge freedom) - use tight bounds
epsilon = 1e-9
for i in range(6):
    lower_bounds[i] = x0[i] - epsilon
    upper_bounds[i] = x0[i] + epsilon

# Fix camera 0 depth scale to 1.0
lower_bounds[n_cameras*6] = 1.0 - epsilon
upper_bounds[n_cameras*6] = 1.0 + epsilon

# Constrain other depth scales to reasonable range [0.1, 10]
for i in range(1, n_cameras):
    lower_bounds[n_cameras*6 + i] = 0.1
    upper_bounds[n_cameras*6 + i] = 15.0

bounds = (lower_bounds, upper_bounds)

# --- RUN OPTIMIZATION ---
print("\n=== Running Optimization ===")
print("This may take a few minutes...\n")

result = least_squares(
    residuals,
    x0,
    bounds=bounds,
    args=(constraints, K),
    verbose=2,
    max_nfev=50,  # Max iterations
    ftol=1e-6,
    xtol=1e-6,
    method='trf'  # Trust Region Reflective algorithm
)

print("\n=== Optimization Complete ===")
print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"Initial cost: {0.5 * (result.fun[0]**2).sum():.6f}")
print(f"Final cost: {result.cost:.6f}")
print(f"Cost reduction: {(1 - result.cost / (0.5 * (result.fun[0]**2).sum())) * 100:.1f}%")
print(f"Iterations: {result.nfev}")

# --- EXTRACT RESULTS ---
print("\n=== Extracting Results ===")

# Extract optimized poses
optimized_poses = []
for i in range(n_cameras):
    pose_vec = result.x[i*6:(i+1)*6]
    optimized_poses.append(vec_to_pose(pose_vec))

optimized_poses = np.array(optimized_poses)

# Extract optimized depth scales
optimized_depth_scales = result.x[n_cameras*6:]

print("\nOptimized Depth Scales:")
for i, scale in enumerate(optimized_depth_scales):
    print(f"  Camera {i:02d}: {scale:.4f}")

print("\nDepth scale statistics:")
print(f"  Min: {optimized_depth_scales.min():.4f}")
print(f"  Max: {optimized_depth_scales.max():.4f}")
print(f"  Mean: {optimized_depth_scales.mean():.4f}")
print(f"  Median: {np.median(optimized_depth_scales):.4f}")
print(f"  Std: {optimized_depth_scales.std():.4f}")

# Check for outliers
outliers = np.abs(optimized_depth_scales - np.median(optimized_depth_scales)) > 2 * optimized_depth_scales.std()
if outliers.any():
    print(f"\nWarning: {outliers.sum()} potential outlier scales detected:")
    for i in np.where(outliers)[0]:
        print(f"  Camera {i}: {optimized_depth_scales[i]:.4f}")

# --- SAVE RESULTS ---
np.save(f"{OUTPUT_DIR}/optimized_poses.npy", optimized_poses)
np.save(f"{OUTPUT_DIR}/optimized_depth_scales.npy", optimized_depth_scales)

print(f"\nSaved results to {OUTPUT_DIR}/")
print("  - optimized_poses.npy")
print("  - optimized_depth_scales.npy")

# --- VISUALIZATION ---
print("\n=== Creating Visualizations ===")

fig = plt.figure(figsize=(18, 5))

# Initial trajectory
ax1 = fig.add_subplot(131, projection='3d')
initial_positions = initial_poses[:, :3, 3]
ax1.plot(initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2], 
         'b.-', linewidth=2, markersize=8, label='Initial', alpha=0.7)
ax1.scatter(initial_positions[0, 0], initial_positions[0, 1], initial_positions[0, 2],
            c='g', s=200, marker='*', label='Start', edgecolors='black', linewidths=2)
ax1.set_xlabel('X (m)', fontsize=10)
ax1.set_ylabel('Y (m)', fontsize=10)
ax1.set_zlabel('Z (m)', fontsize=10)
ax1.set_title('Initial Camera Trajectory\n(from recoverPose)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Optimized trajectory
ax2 = fig.add_subplot(132, projection='3d')
optimized_positions = optimized_poses[:, :3, 3]
ax2.plot(optimized_positions[:, 0], optimized_positions[:, 1], optimized_positions[:, 2],
         'r.-', linewidth=2, markersize=8, label='Optimized', alpha=0.7)
ax2.scatter(optimized_positions[0, 0], optimized_positions[0, 1], optimized_positions[0, 2],
            c='g', s=200, marker='*', label='Start', edgecolors='black', linewidths=2)
ax2.set_xlabel('X (m)', fontsize=10)
ax2.set_ylabel('Y (m)', fontsize=10)
ax2.set_zlabel('Z (m)', fontsize=10)
ax2.set_title('Optimized Camera Trajectory\n(after depth consistency)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Depth scales
ax3 = fig.add_subplot(133)
colors = ['red' if outliers[i] else 'steelblue' for i in range(n_cameras)]
ax3.bar(range(n_cameras), optimized_depth_scales, color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(y=1.0, color='darkgreen', linestyle='--', linewidth=2, label='Reference (1.0)', alpha=0.7)
ax3.axhline(y=np.median(optimized_depth_scales), color='orange', linestyle=':', linewidth=2, 
            label=f'Median ({np.median(optimized_depth_scales):.3f})', alpha=0.7)
ax3.set_xlabel('Camera Index', fontsize=12)
ax3.set_ylabel('Depth Scale Factor', fontsize=12)
ax3.set_title('Optimized Depth Scales\n(Red = potential outliers)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(range(0, n_cameras, max(1, n_cameras//10)))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/optimization_results.png", dpi=150, bbox_inches='tight')
print(f"Saved visualization to {OUTPUT_DIR}/optimization_results.png")

# Additional: Plot trajectory comparison overlay
fig2, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': '3d'})
ax.plot(initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2], 
        'b.-', linewidth=2, markersize=6, label='Initial', alpha=0.5)
ax.plot(optimized_positions[:, 0], optimized_positions[:, 1], optimized_positions[:, 2],
        'r.-', linewidth=2, markersize=6, label='Optimized', alpha=0.7)
ax.scatter(initial_positions[0, 0], initial_positions[0, 1], initial_positions[0, 2],
           c='g', s=300, marker='*', label='Start', edgecolors='black', linewidths=2, zorder=10)
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.set_title('Camera Trajectory Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/trajectory_comparison.png", dpi=150, bbox_inches='tight')
print(f"Saved trajectory comparison to {OUTPUT_DIR}/trajectory_comparison.png")

print("\n=== Done! ===")
print("Next step: Run TSDF fusion with optimized poses and depth scales")


# After optimization completes:
print("\n=== Optimization Diagnostics ===")
print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"Optimization method: {result.method if hasattr(result, 'method') else 'N/A'}")
print(f"Number of function evaluations: {result.nfev}")
print(f"Number of Jacobian evaluations: {result.njev if hasattr(result, 'njev') else 'N/A'}")

print(f"\nCost:")
print(f"  Initial cost: {0.5 * np.sum(residuals(x0, constraints, K)**2):.6f}")
print(f"  Final cost: {result.cost:.6f}")
print(f"  Cost reduction: {((0.5 * np.sum(residuals(x0, constraints, K)**2) - result.cost) / (0.5 * np.sum(residuals(x0, constraints, K)**2) * 100)):.2f}%")

print(f"\nResiduals:")
initial_residuals = residuals(x0, constraints, K)
final_residuals = result.fun
print(f"  Initial RMS error: {np.sqrt(np.mean(initial_residuals**2)):.6f}")
print(f"  Final RMS error: {np.sqrt(np.mean(final_residuals**2)):.6f}")
print(f"  Initial max error: {np.abs(initial_residuals).max():.6f}")
print(f"  Final max error: {np.abs(final_residuals).max():.6f}")

print(f"\nParameter changes:")
print(f"  Max pose change: {np.abs(result.x[:n_cameras*6] - x0[:n_cameras*6]).max():.6f}")
print(f"  Max scale change: {np.abs(result.x[n_cameras*6:] - x0[n_cameras*6:]).max():.6f}")