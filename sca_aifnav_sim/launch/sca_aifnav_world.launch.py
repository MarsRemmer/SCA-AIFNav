"""Launch the SCA-AIFNav Gazebo simulation environment."""

import os

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import (
    LaunchConfiguration,
)
from launch_ros.actions import Node


def generate_launch_description():
    """Create the Gazebo world and spawn the navigation robot."""
    sim_share = get_package_share_directory(
        "sca_aifnav_sim"
    )

    gazebo_share = get_package_share_directory(
        "gazebo_ros"
    )

    turtlebot3_gazebo_share = (
        get_package_share_directory(
            "turtlebot3_gazebo"
        )
    )

    model_file = os.path.join(
        sim_share,
        "models",
        "sca_aifnav_waffle_pi",
        "model.sdf",
    )

    default_world = os.path.join(
        turtlebot3_gazebo_share,
        "worlds",
        "turtlebot3_world.world",
    )

    model_paths = [
        os.path.join(
            sim_share,
            "models",
        ),
        os.path.join(
            turtlebot3_gazebo_share,
            "models",
        ),
    ]

    existing_model_path = os.environ.get(
        "GAZEBO_MODEL_PATH",
        "",
    )

    if existing_model_path:
        model_paths.append(
            existing_model_path
        )

    gazebo_model_path = os.pathsep.join(
        model_paths
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

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                gazebo_share,
                "launch",
                "gazebo.launch.py",
            )
        ),
        launch_arguments={
            "world": world,
        }.items(),
    )

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_sca_aifnav_robot",
        output="screen",
        arguments=[
            "-entity",
            "sca_aifnav_waffle_pi",
            "-file",
            model_file,
            "-x",
            spawn_x,
            "-y",
            spawn_y,
            "-z",
            spawn_z,
            "-Y",
            spawn_yaw,
        ],
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                name="GAZEBO_MODEL_PATH",
                value=gazebo_model_path,
            ),
            SetEnvironmentVariable(
                name="TURTLEBOT3_MODEL",
                value="waffle_pi",
            ),
            DeclareLaunchArgument(
                "world",
                default_value=default_world,
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
            gazebo,
            spawn_robot,
        ]
    )
