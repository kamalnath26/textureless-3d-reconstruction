from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    # Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    video_path = LaunchConfiguration('video_path', default='')
    
    declare_video_path_arg = DeclareLaunchArgument(
        'video_path',
        default_value='',
        description='Path to video file for playback (optional)'
    )

    declare_framerate_arg = DeclareLaunchArgument(
        'framerate',
        default_value='30.0',
        description='Framerate for camera or video playback'
    )

    declare_db_path_arg = DeclareLaunchArgument(
        'db_path',
        default_value='',
        description='Path to RTAB-Map database file for playback'
    )

    video_path_config = LaunchConfiguration('video_path')
    framerate_config = LaunchConfiguration('framerate')
    db_path_config = LaunchConfiguration('db_path')

    # Conditional logic to choose between simple_camera_node and db_player_node
    # We can't easily use Python conditionals with LaunchConfiguration values at parse time
    # So we'll launch both but use a condition to only start one.
    # Actually, it's cleaner to use a Python function or just assume if db_path is set, we use it.
    
    from launch.conditions import IfCondition, UnlessCondition
    from launch.substitutions import PythonExpression

    # Condition: Use DB player if db_path is NOT empty
    use_db_player = PythonExpression(["'", db_path_config, "' != ''"])
    # Condition: Use Camera/Video if db_path IS empty
    use_camera = PythonExpression(["'", db_path_config, "' == ''"])

    # 1. Camera Node (simple_camera_node - OpenCV based)
    camera_node = Node(
        package='monocular_slam',
        executable='simple_camera_node',
        name='simple_camera_node',
        parameters=[{
            'video_path': video_path_config,
            'framerate': framerate_config,
            'frame_id': 'camera_optical_frame'
        }],
        condition=IfCondition(use_camera)
    )

    # 1.5. DB Player Node (for playing back RTAB-Map databases)
    db_player_node = Node(
        package='monocular_slam',
        executable='db_player_node',
        name='db_player_node',
        parameters=[{
            'db_path': db_path_config,
            'framerate': framerate_config,
            'frame_id': 'camera_optical_frame'
        }],
        condition=IfCondition(use_db_player)
    )
    
    # Nodes
    
    # 2. Depth Anything Node
    depth_anything_node = Node(
        package='monocular_slam',
        executable='depth_anything_node',
        name='depth_anything_node',
        output='screen',
        parameters=[{
            'model_id': 'LiheYoung/depth-anything-base-hf',
            # 'compute_device': 'cuda' # Optional, defaults to auto-detect
        }]
    )
    
    # 3. RTAB-Map
    # We use rtabmap_ros rgbd_odometry and rtabmap nodes
    
    rtabmap_args = {
        'frame_id': 'camera',
        'subscribe_depth': True,
        'subscribe_rgb': True,
        'approx_sync': True, # Important since depth is generated with slight delay
        'wait_for_transform': 0.5, # Wait for TF
        'sync_queue_size': 100, # Large queue to buffer RGB while waiting for slow depth
        'topic_queue_size': 10,
        'qos': 2, # Reliability: 1=Reliable, 2=Best Effort. Camera drivers often use Best Effort.
    }
    
    # RTAB-Map Odometry
    rgbd_odometry_node = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        output='screen',
        parameters=[rtabmap_args, {
            'Odom/Strategy': '1', # 0=Frame-to-Map (F2M) 1=Frame-to-Frame (F2F)
            'Odom/ResetCountdown': '1',
            'Reg/Force3DoF': 'false',
            'Vis/MinInliers': '15', # Lower inliers threshold (default 20)
            'Vis/CorType': '1', # 0=Features Matching, 1=Optical Flow (often smoother for video)
            'GFTT/MinDistance': '10', # Extract more features
        }],
        remappings=[
            ('rgb/image', '/camera/image_raw'),
            ('depth/image', '/camera/depth_registered/image_raw'),
            ('rgb/camera_info', '/camera/camera_info'),
            ('odom', '/odom'),
        ]
    )
    
    # RTAB-Map SLAM
    rtabmap_slam_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        output='screen',
        parameters=[rtabmap_args, {
            'Rtabmap/DetectionRate': '1',
            'Mem/IncrementalMemory': 'true',
            'Mem/InitWMWithAllNodes': 'false',
            'Grid/RangeMax': '5.0', # Limit 2D map to 5m
            'Grid/DepthMax': '5.0', # Limit 3D cloud to 5m (filters noisy far points)
            # 'Vis/MaxDepth': '0', # 0=Inf. Keep this high/unlimited for robust odometry!
        }],
        remappings=[
            ('rgb/image', '/camera/image_raw'),
            ('depth/image', '/camera/depth_registered/image_raw'),
            ('rgb/camera_info', '/camera/camera_info'),
            ('odom', '/odom'),
        ],
        arguments=['--delete_db_on_start']
    )
    
    # Visualization (RTAB-Map Viz)
    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        output='screen',
        parameters=[rtabmap_args],
        remappings=[
            ('rgb/image', '/camera/image_raw'),
            ('depth/image', '/camera/depth_registered/image_raw'),
            ('rgb/camera_info', '/camera/camera_info'),
            ('odom', '/odom'),
        ]
    )
    
    # Static TF (base_link -> camera)
    # Assuming camera is at 0,0,0 relative to base_link for simplicity
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'camera'] 
    )

    # Static TF (camera -> camera_optical_frame)
    # Transform from standard ROS camera frame (X-forward) to Optical frame (Z-forward)
    # yaw=-1.57 (right), pitch=0, roll=-1.57 (down) -> This aligns X-forward to Z-forward
    static_tf_node_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '-1.5708', '0', '-1.5708', 'camera', 'camera_optical_frame']
    )

    return LaunchDescription([
        declare_video_path_arg,
        declare_framerate_arg,
        declare_db_path_arg,
        static_tf_node,
        static_tf_node_optical,
        camera_node,
        db_player_node,
        depth_anything_node,
        rgbd_odometry_node,
        rtabmap_slam_node,
        rtabmap_viz_node
    ])
