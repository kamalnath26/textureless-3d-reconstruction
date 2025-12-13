## Sample commands to use reconstruction.py

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

<<<<<<< HEAD
## exmaple usage of depth_enhanced_reconstruction.py

```bash
python depth_enhanced_reconstruction.py --input ./input_folder/buddha_images --output ./output --fx 1719 --fy 1719 --cx 540 --cy 960
```

## exmaple usage of depth_to_reconstruction.py

```bash
python depth_to_reconstruction.py --rgb-folder ./input_folder/buddha_images --depth-folder ./depth_output/depth_images --output ./reconstruction.ply --fx 1719 --fy 1719 --cx 540 --cy 960
```

refer to top of pyhton script and the main function for more args that supported by these scripts

link for dataset collected: [link](https://northeastern-my.sharepoint.com/my?id=%2Fpersonal%2Fbathirappan%5Fk%5Fnortheastern%5Fedu%2FDocuments%2FAFR%20Project%2Fdataset&viewid=f174bb69%2D4814%2D4f8b%2Db313%2Dbd3ab70a5678&login_hint=bathirappan%2Ek%40northeastern%2Eedu&source=waffle)

link for outputs:[link](https://northeastern-my.sharepoint.com/my?id=%2Fpersonal%2Fbathirappan%5Fk%5Fnortheastern%5Fedu%2FDocuments%2FAFR%20Project%2Foutput%5Ffolder&viewid=f174bb69%2D4814%2D4f8b%2Db313%2Dbd3ab70a5678&login_hint=bathirappan%2Ek%40northeastern%2Eedu&source=waffle)

<video  controls>
  <source src="./output_folder/reconstruction_buddha_derc.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<video  controls>
  <source src="./output_folder/exp_no_feature_data_left_to_right_derc_pcl.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<video  controls>
  <source src="./output_folder/exp_tunnel_set_1_derc_pcl.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
=======
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
>>>>>>> 080c28e0cc703cbb7db0c9ad59db66d0fa8ee787
