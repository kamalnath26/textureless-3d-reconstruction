import open3d as o3d
import numpy as np
import cv2
import os
import json
from tqdm import tqdm

# --- CONFIG ---
RGB_DIR = "images_resized"
DEPTH_DIR = "depth_resized"
OUTPUT_DIR = "loftr_output"
INTRINSICS_FILE = "intrinsics.json"
DEPTH_RANGES_FILE = "per_image_depth_ranges.json"

# TSDF parameters
VOXEL_LENGTH = 0.01  # 1cm voxels - adjust based on scene size
SDF_TRUNC = 0.04     # 4cm truncation distance
MAX_DEPTH = 10.0     # Maximum depth to integrate

# --- LOAD DATA ---
print("=== Loading Optimized Data ===")

# Load optimized poses
poses = np.load(f"{OUTPUT_DIR}/optimized_poses.npy")
print(f"Loaded {len(poses)} optimized camera poses")

# Load optimized depth scales
depth_scales = np.load(f"{OUTPUT_DIR}/optimized_depth_scales.npy")
print(f"Loaded {len(depth_scales)} optimized depth scales")

print(f"\nDepth scales: min={depth_scales.min():.3f}, max={depth_scales.max():.3f}, mean={depth_scales.mean():.3f}")


# Load results


print(f"\nIndividual scales:")
for i, s in enumerate(depth_scales):
    print(f"  Camera {i:02d}: {s:.4f}")

print(f"\nPose translations (camera positions):")
for i in range(min(5, len(poses))):
    t = poses[i][:3, 3]
    print(f"  Camera {i:02d}: [{t[0]:7.3f}, {t[1]:7.3f}, {t[2]:7.3f}]")
# Load intrinsics
with open(INTRINSICS_FILE, "r") as f:
    intrinsics_dict = json.load(f)

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(
    intrinsics_dict["width"],
    intrinsics_dict["height"],
    intrinsics_dict["fx"],
    intrinsics_dict["fy"],
    intrinsics_dict["cx"],
    intrinsics_dict["cy"]
)

print(f"\nIntrinsics: {intrinsics_dict['width']}x{intrinsics_dict['height']}")
print(f"  fx={intrinsics_dict['fx']:.2f}, fy={intrinsics_dict['fy']:.2f}")
print(f"  cx={intrinsics_dict['cx']:.2f}, cy={intrinsics_dict['cy']:.2f}")

# Load depth ranges
with open(DEPTH_RANGES_FILE, "r") as f:
    depth_ranges = json.load(f)

depth_lookup = {item['img_id']: item for item in depth_ranges}


# --- LOAD AND PROCESS IMAGES ---
print("\n=== Loading Images and Depth Maps ===")

rgb_files = sorted(os.listdir(RGB_DIR))
n_images = min(len(rgb_files), len(poses))

print(f"Processing {n_images} images")

# --- INITIALIZE TSDF VOLUME ---
print("\n=== Initializing TSDF Volume ===")

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=VOXEL_LENGTH,
    sdf_trunc=SDF_TRUNC,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

print(f"TSDF Parameters:")
print(f"  Voxel length: {VOXEL_LENGTH}m ({VOXEL_LENGTH*1000:.1f}mm)")
print(f"  SDF truncation: {SDF_TRUNC}m ({SDF_TRUNC*1000:.1f}mm)")
print(f"  Max depth: {MAX_DEPTH}m")

# --- INTEGRATE FRAMES ---
print("\n=== Integrating Frames into TSDF ===")

integrated_count = 0
skipped_count = 0

for i in tqdm(range(n_images), desc="TSDF Integration"):
    # Load RGB image
    rgb_path = os.path.join(RGB_DIR, rgb_files[i])
    rgb = cv2.imread(rgb_path)
    
    if rgb is None:
        print(f"\nWarning: Cannot load RGB image {rgb_files[i]}")
        skipped_count += 1
        continue
    
    # Load depth map (RGB format)
    depth_path = os.path.join(DEPTH_DIR, rgb_files[i])
    depth_rgb = cv2.imread(depth_path)  # Load RGB depth
    
    if depth_rgb is None:
        print(f"\nWarning: Cannot load depth map {rgb_files[i]}")
        skipped_count += 1
        continue
    
    # Extract green channel (most informative)
    depth_raw = depth_rgb[:, :, 1]
    
    # Ensure same size
    h, w = rgb.shape[:2]
    if depth_raw.shape != (h, w):
        depth_raw = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # --- PROCESS DEPTH MAP ---
    # Normalize to [0, 1]
    depth_normalized = depth_raw.astype(np.float32) / 255.0
    
    # Map to relative depth using per-image range
    if i in depth_lookup:
        depth_info = depth_lookup[i]
        min_d = depth_info['min']
        max_d = depth_info['max']
        depth_relative = depth_normalized * (max_d - min_d) + min_d
    else:
        print(f"\nWarning: No depth range for image {i}, using normalized depth")
        depth_relative = depth_normalized
    
    # Apply optimized scale factor
    depth_metric = depth_relative * depth_scales[i]
    depth_metric = np.max(depth_metric) - depth_metric + 0.1  # Invert depth
    
    # Clip to valid range
    depth_metric = np.clip(depth_metric, 0.01, MAX_DEPTH)
    
    # --- CREATE RGBD IMAGE ---
    # Convert RGB to Open3D format
    rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.uint8))
    
    # Convert depth to Open3D format
    depth_o3d = o3d.geometry.Image(depth_metric.astype(np.float32))
    
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1.0,  # Already in meters
        depth_trunc=MAX_DEPTH,
        convert_rgb_to_intensity=False
    )
    
    # --- GET CAMERA POSE ---
    # Open3D uses world-to-camera (extrinsic), we have camera-to-world
    # So we need to invert
    extrinsic = poses[i]
    
    # Sanity check
    if not np.isfinite(extrinsic).all():
        print(f"\nWarning: Invalid pose for image {i}, skipping")
        skipped_count += 1
        continue
    
    # --- INTEGRATE ---
    try:
        volume.integrate(rgbd, intrinsic, extrinsic)
        integrated_count += 1
    except Exception as e:
        print(f"\nError integrating frame {i}: {e}")
        skipped_count += 1
        continue

print(f"\n=== Integration Complete ===")
print(f"Successfully integrated: {integrated_count}/{n_images} frames")
print(f"Skipped: {skipped_count} frames")

# --- EXTRACT MESH ---
print("\n=== Extracting Mesh ===")

mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()

print(f"Mesh statistics:")
print(f"  Vertices: {len(mesh.vertices)}")
print(f"  Triangles: {len(mesh.triangles)}")

if len(mesh.vertices) == 0:
    print("\nERROR: Mesh has no vertices!")
    print("Possible issues:")
    print("  - Depth scales might be wrong")
    print("  - Camera poses might be incorrect")
    print("  - TSDF parameters might need adjustment")
    exit(1)

# --- OPTIONAL: CLEAN UP MESH ---
# --- OPTIONAL: CLEAN UP MESH ---
print("\n=== Cleaning Mesh ===")

# Option 1: Simple cleanup without statistical outlier removal
print("Removing degenerate triangles...")
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_non_manifold_edges()

# Option 2: Remove small disconnected components
print("Removing small components...")
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh.cluster_connected_triangles())

triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)

if len(cluster_n_triangles) > 0:
    # Keep only the largest component
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(f"  Kept largest component: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
else:
    print("  No clustering needed")

# Optionally: smooth the mesh slightly
# mesh = mesh.filter_smooth_simple(number_of_iterations=1)
# --- SAVE MESH ---
output_mesh_path = f"{OUTPUT_DIR}/reconstruction_mesh.ply"
o3d.io.write_triangle_mesh(output_mesh_path, mesh)
print(f"\n=== Saved Mesh ===")
print(f"Output: {output_mesh_path}")

# --- VISUALIZE ---
print("\n=== Visualizing Result ===")
print("Close the window to continue...")

# Create coordinate frame at origin
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# Add camera frustums for visualization
camera_frustums = []
for i in range(0, n_images, max(1, n_images//10)):  # Show every Nth camera
    frustum = o3d.geometry.LineSet.create_camera_visualization(
        intrinsic.intrinsic_matrix,
        np.linalg.inv(poses[i]),  # World-to-camera for visualization
        scale=0.05
    )
    frustum.paint_uniform_color([1, 0, 0])  # Red frustums
    camera_frustums.append(frustum)

# Visualize
o3d.visualization.draw_geometries(
    [mesh, coord_frame] + camera_frustums,
    window_name="3D Reconstruction",
    width=1920,
    height=1080,
    left=50,
    top=50
)

print("\n=== Done! ===")
print(f"Final mesh saved to: {output_mesh_path}")
print(f"  Vertices: {len(mesh.vertices)}")
print(f"  Triangles: {len(mesh.triangles)}")