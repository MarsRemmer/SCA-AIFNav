"""Launch Gazebo, odom-only Nav2, and the SCA-AIFNav navigation node."""

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
    """Create the complete SCA-AIFNav simulation and navigation stack."""
    sim_share = get_package_share_directory(
        "sca_aifnav_sim"
    )

    nav2_share = get_package_share_directory(
        "nav2_bringup"
    )

    nav2_params = os.path.join(
        sim_share,
        "config",
        "nav2_odom_params.yaml",
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

    base_frame_id = LaunchConfiguration(
        "base_frame_id"
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

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                nav2_share,
                "launch",
                "navigation_launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": "true",
            "autostart": "true",
            "params_file": nav2_params,
            "use_composition": "False",
            "use_respawn": "False",
        }.items(),
    )

    navigation_node = Node(
        package="sca_aifnav_ros",
        executable="navigation_node",
        name="sca_aifnav_navigation",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "odom_topic": "/odom",
                "agent_odom_topic": "/agent/odom",
                "scan_topic": "/scan",
                "camera_topic": "/camera_front/image_raw",
                "left_camera_topic": "/camera_left/image_raw",
                "right_camera_topic": "/camera_right/image_raw",

                # With Nav2, both controller_server and SCA panorama
                # commands feed the Nav2 velocity smoother.  The smoother
                # is the only component that publishes final /cmd_vel.
                "cmd_vel_topic": "/cmd_vel_nav",

                "navigation_motion_backend": "nav2",

                "base_frame_id": (
                    base_frame_id
                ),

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
                "base_frame_id",
                default_value="base_link",
                description=(
                    "Robot base frame used for sensor TF lookup."
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
            nav2_launch,
            navigation_node,
        ]
    )
