## Sample commands to use

From a folder of images:

```bash
python reconstruction.py --mode folder --input ./my_images/ --output scene.ply
```

From webcam (real-time SLAM-like):

```bash
python reconstruction.py --mode camera --camera 0 --output scene.ply
```

Using USB camera:

```bash
python reconstruction.py --mode camera --camera 1 --output scene.ply
```

## sample commands to use depth_process.py

From images folder, output depth images only
```bash
python depth_processor.py --input ./images --output ./output --mode images
```

From USB camera, 1 fps, output both depth and pointcloud
```bash
python depth_processor.py --source camera --device-id 0 --fps-mode 1fps --mode both
```

From video, 50% of frames, with ROS2 publishing at 10Hz
```bash
python depth_processor.py --source video --video-path video.mp4 \
    --fps-mode custom --fps-percent 50 \
    --ros2 --ros2-freq 10 --mode both
```

Using V3 metric model for outdoor scenes
```bash
python depth_processor.py --version v3 --metric --max-depth 80 --dataset vkitti \
    --input ./images --output ./output --mode both
```

With custom camera intrinsics (for RealSense D455)
```bash
python depth_processor.py --source camera --intrinsics camera_intrinsics.json \
    --mode both --ros2 --preview
```

there are so many options refer to main function in the code to use more combinations of input or output

## ROS 2 Monocular SLAM Pipeline

Make sure to source your workspace first:
```bash
cd ros2_ws
source install/setup.bash
```

### 1. Run with a Video File
This uses `simple_camera_node` to play a video file in a loop.
```bash
ros2 launch monocular_slam slam.launch.py video_path:=/absolute/path/to/your_video.mp4
```

### 2. Run with a Database (RTAB-Map DB)
This uses `db_player_node` to replay images and calibration from a recorded database.
```bash
ros2 launch monocular_slam slam.launch.py db_path:=/absolute/path/to/your_database.db
```

### 3. Run with Live Webcam
This uses `simple_camera_node` with the default video device (usually `/dev/video0`).
```bash
ros2 launch monocular_slam slam.launch.py
```