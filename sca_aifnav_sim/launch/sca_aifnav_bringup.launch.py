"""Launch the complete SCA-AIFNav simulation and navigation stack."""

import os

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import (
    LaunchConfiguration,
)
from launch_ros.actions import Node


def generate_launch_description():
    """Create the complete simulation and autonomous navigation stack."""
    sim_share = get_package_share_directory(
        "sca_aifnav_sim"
    )

    world = LaunchConfiguration(
        "world"
    )

    spawn_x = LaunchConfiguration(
        "x"
    )

    spawn_y = LaunchConfiguration(
        "y"
    )

    spawn_z = LaunchConfiguration(
        "z"
    )

    spawn_yaw = LaunchConfiguration(
        "yaw"
    )

    laser_yaw_offset_rad = LaunchConfiguration(
        "laser_yaw_offset_rad"
    )

    panorama_control_period_sec = LaunchConfiguration(
        "panorama_control_period_sec"
    )

    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                sim_share,
                "launch",
                "sca_aifnav_world.launch.py",
            )
        ),
        launch_arguments={
            "world": world,
            "x": spawn_x,
            "y": spawn_y,
            "z": spawn_z,
            "yaw": spawn_yaw,
        }.items(),
    )

    navigation_node = Node(
        package="sca_aifnav_ros",
        executable="navigation_node",
        name="sca_aifnav_navigation",
        output="screen",
        parameters=[
            {
                "odom_topic": "/odom",
                "scan_topic": "/scan",
                "camera_topic": "/camera_front/image_raw",
                "left_camera_topic": "/camera_left/image_raw",
                "right_camera_topic": "/camera_right/image_raw",
                "cmd_vel_topic": "/cmd_vel",
                "laser_yaw_offset_rad": laser_yaw_offset_rad,
                "panorama_control_period_sec": (
                    panorama_control_period_sec
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world",
                default_value=os.path.join(
                    get_package_share_directory(
                        "turtlebot3_gazebo"
                    ),
                    "worlds",
                    "turtlebot3_world.world",
                ),
                description=(
                    "Gazebo world file to load."
                ),
            ),
            DeclareLaunchArgument(
                "x",
                default_value="-2.0",
                description=(
                    "Initial robot x position."
                ),
            ),
            DeclareLaunchArgument(
                "y",
                default_value="-0.5",
                description=(
                    "Initial robot y position."
                ),
            ),
            DeclareLaunchArgument(
                "z",
                default_value="0.01",
                description=(
                    "Initial robot z position."
                ),
            ),
            DeclareLaunchArgument(
                "yaw",
                default_value="0.0",
                description=(
                    "Initial robot yaw."
                ),
            ),
            DeclareLaunchArgument(
                "laser_yaw_offset_rad",
                default_value="0.0",
                description=(
                    "Yaw offset from robot frame to laser frame."
                ),
            ),
            DeclareLaunchArgument(
                "panorama_control_period_sec",
                default_value="1.0",
                description=(
                    "Panorama control-loop period in seconds."
                ),
            ),
            world_launch,
            navigation_node,
        ]
    )
