import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/garrett/VSCode/EECE7150-final2/ros2_ws/install/monocular_slam'
