from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'monocular_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='garrett',
    maintainer_email='garrett@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simple_camera_node = monocular_slam.simple_camera_node:main',
            'depth_anything_node = monocular_slam.depth_anything_node:main',
            'db_player_node = monocular_slam.db_player_node:main',
            'droid_slam_node = monocular_slam.droid_slam_node:main',
        ],
    },
)
