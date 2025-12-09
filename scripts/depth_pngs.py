import cv2
import os

depth_dir = "./depth/"               # raw depth outputs (side-by-side images)
rgb_dir = "./images_resized/"                # your true RGBs
out_dir = "depth_resized"           # cropped and resized depth maps
os.makedirs(out_dir, exist_ok=True)

# Read sample rgb to get target size
rgb_sample = cv2.imread(os.path.join(rgb_dir, sorted(os.listdir(rgb_dir))[0]))
H, W = rgb_sample.shape[:2]
print("Target RGB size:", W, H)

depth_files = sorted(os.listdir(depth_dir))

for fname in depth_files:
    path = os.path.join(depth_dir, fname)
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h, w = d.shape[:2]
    mid = w // 2  # assume exact left=rgb, right=depth

    depth_crop = d[:, mid:]   # take RIGHT HALF only

    # Resize depth to match RGB resolution
    depth_resized = cv2.resize(depth_crop, (W, H), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(out_dir, fname), depth_resized)

print("Cropped & resized all depth images!")

# depth_test = cv2.imread("depth_resized/buddha_010.png", cv2.IMREAD_UNCHANGED)
# print(f"Depth dtype: {depth_test.dtype}")
# print(f"Depth range: [{depth_test.min()}, {depth_test.max()}]")