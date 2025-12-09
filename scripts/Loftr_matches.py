import cv2
import numpy as np
import torch
from kornia.feature import LoFTR
from pathlib import Path
import json
from tqdm import tqdm
import gc

# --- CONFIG ---
IMAGE_DIR = "images_resized"
OUTPUT_DIR = "loftr_output"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Device with memory management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Clear any existing cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# Load LOFTR with lower resolution settings
matcher = LoFTR(pretrained='outdoor').to(device).eval()

# CRITICAL: Set lower resolution for LOFTR coarse matching
matcher.config['resolution'] = (8, 2)
matcher.config['coarse']['d_model'] = 256
matcher.config['coarse']['nhead'] = 8

# Load intrinsics
with open("intrinsics.json", "r") as f:
    intrinsics = json.load(f)
    
K = np.array([
    [intrinsics['fx'], 0, intrinsics['cx']],
    [0, intrinsics['fy'], intrinsics['cy']],
    [0, 0, 1]
])

print(f"Intrinsics K:\n{K}")

# --- LOAD IMAGE PATHS ---
image_files = sorted(Path(IMAGE_DIR).glob("*.png"))
print(f"Found {len(image_files)} images")

# --- HELPER FUNCTIONS ---
def preprocess_image(img, max_dim=840):
    """Convert to grayscale tensor and optionally downsample for matching only"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    gray = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(gray)[None, None].to(device)
    
    return tensor, scale

def get_loftr_matches(img1_path, img2_path):
    """Get LOFTR matches between two images"""
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        return None, None, None
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    img1_tensor, scale1 = preprocess_image(img1, max_dim=840)
    img2_tensor, scale2 = preprocess_image(img2, max_dim=840)
    
    input_dict = {
        'image0': img1_tensor,
        'image1': img2_tensor
    }
    
    with torch.no_grad():
        try:
            correspondences = matcher(input_dict)
        except RuntimeError as e:
            print(f"    LOFTR failed: {e}")
            return None, None, None
    
    # Scale back to original resolution
    mkpts0 = correspondences['keypoints0'].cpu().numpy() / scale1
    mkpts1 = correspondences['keypoints1'].cpu().numpy() / scale2
    confidence = correspondences['confidence'].cpu().numpy()
    
    del img1, img2, img1_tensor, img2_tensor, correspondences, input_dict
    torch.cuda.empty_cache()
    gc.collect()
    
    return mkpts0, mkpts1, confidence

def filter_matches_by_confidence(mkpts0, mkpts1, confidence, threshold=0.5):
    """Keep only high-confidence matches"""
    if mkpts0 is None:
        return None, None, None
    mask = confidence > threshold
    return mkpts0[mask], mkpts1[mask], confidence[mask]

# --- EXTRACT MATCHES ---
print("\n=== Extracting LOFTR Matches ===")

all_matches = {}

# Consecutive pairs + loop closures
pairs_to_match = [(i, i+1) for i in range(len(image_files) - 1)]

for i in range(0, len(image_files), 6):
    for j in range(i+6, len(image_files), 6):
        pairs_to_match.append((i, j))

if len(image_files) > 5:
    pairs_to_match.append((0, len(image_files)-1))

print(f"Matching {len(pairs_to_match)} pairs")

for i, j in tqdm(pairs_to_match, desc="Running LOFTR"):
    try:
        mkpts0, mkpts1, conf = get_loftr_matches(image_files[i], image_files[j])
        
        if mkpts0 is None:
            print(f"  Pair ({i:02d}, {j:02d}): Failed")
            continue
        
        mkpts0, mkpts1, conf = filter_matches_by_confidence(mkpts0, mkpts1, conf, threshold=0.5)
        
        if mkpts0 is None or len(mkpts0) < 30:
            print(f"  Pair ({i:02d}, {j:02d}): Too few matches ({len(mkpts0) if mkpts0 is not None else 0})")
            continue
        
        print(f"  Pair ({i:02d}, {j:02d}): {len(mkpts0)} matches ✓")
        all_matches[(i, j)] = (mkpts0, mkpts1, conf)
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  Pair ({i:02d}, {j:02d}): OOM - skipping")
        torch.cuda.empty_cache()
        gc.collect()
        continue
    except Exception as e:
        print(f"  Pair ({i:02d}, {j:02d}): Error - {e}")
        continue

print(f"\nSuccessfully matched {len(all_matches)} pairs")

# --- SAVE MATCHES (ONLY ONE PLACE!) ---
if len(all_matches) > 0:
    match_data = {}
    for (i, j), (mkpts0, mkpts1, conf) in all_matches.items():
        match_data[f"pair_{i}_{j}_mkpts0"] = mkpts0
        match_data[f"pair_{i}_{j}_mkpts1"] = mkpts1
        match_data[f"pair_{i}_{j}_conf"] = conf
    
    np.savez(f"{OUTPUT_DIR}/loftr_matches.npz", **match_data)
    print(f"Saved matches to {OUTPUT_DIR}/loftr_matches.npz")
else:
    print("ERROR: No matches found!")
    exit(1)

# --- COMPUTE RELATIVE POSES ---
print("\n=== Computing Relative Poses ===")

relative_poses = {}

for (i, j), (mkpts0, mkpts1, conf) in all_matches.items():
    if len(mkpts0) < 8:
        continue
    
    E, mask = cv2.findEssentialMat(mkpts0, mkpts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    if E is None:
        print(f"  Pair ({i}, {j}): Essential matrix failed")
        continue
    
    n_inliers, R, t, mask_pose = cv2.recoverPose(E, mkpts0, mkpts1, K, mask=mask)
    
    inlier_ratio = n_inliers / len(mkpts0)
    print(f"  Pair ({i:02d}, {j:02d}): {n_inliers}/{len(mkpts0)} inliers ({inlier_ratio:.1%})")
    
    relative_poses[(i, j)] = (R, t.ravel(), mask_pose.ravel())

# --- SAVE RELATIVE POSES ---
pose_data = {}
for (i, j), (R, t, mask) in relative_poses.items():
    pose_data[f"pair_{i}_{j}_R"] = R
    pose_data[f"pair_{i}_{j}_t"] = t
    pose_data[f"pair_{i}_{j}_mask"] = mask

np.savez(f"{OUTPUT_DIR}/relative_poses.npz", **pose_data)
print(f"Saved relative poses to {OUTPUT_DIR}/relative_poses.npz")

# --- BUILD POSE CHAIN ---
print("\n=== Building Pose Chain ===")

def relative_to_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

poses = [np.eye(4)]

for i in range(len(image_files) - 1):
    if (i, i+1) not in relative_poses:
        print(f"WARNING: Missing consecutive pose ({i}, {i+1})")
        poses.append(poses[-1])
        continue
    
    R, t, _ = relative_poses[(i, i+1)]
    relative_T = relative_to_matrix(R, t)
    poses.append(poses[-1] @ relative_T)

poses = np.array(poses)
np.save(f"{OUTPUT_DIR}/initial_poses.npy", poses)

print("\n=== Summary ===")
print(f"Matched pairs: {len(all_matches)}")
print(f"Relative poses: {len(relative_poses)}")
print(f"Pose chain: {len(poses)} cameras")

print("\nCamera positions:")
for i, pose in enumerate(poses[::3]):
    pos = pose[:3, 3]
    print(f"  Camera {i*3:02d}: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}]")