#!/usr/bin/env bash

# SCA-AIFNav Mini Warehouse
# Baseline reproduction - Terminal 5
#
# Shared with AIMAPP:
#   A1 Gazebo server
#   A2 Gazebo GUI
#   A3 waffle_pi_plus + robot_state_publisher
#   A4 AIMAPP Nav2 stack
#
# This script starts only the SCA-AIFNav baseline agent.
# The high-level algorithm remains AIMAPP-aligned.
# Translational navigation is delegated to Nav2.

set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1

source /opt/ros/humble/setup.bash
source "$HOME/SCA-AIFNav-Project/sca_aifnav/runtime_ws/install/setup.bash"

RESULT_ROOT="$HOME/SCA-AIFNav-Project/experiments/mini_warehouse/smoke"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RESULT_ROOT/sca_baseline_nav2_${STAMP}"

mkdir -p "$RUN_DIR"

echo "=========================================="
echo "SCA-AIFNav Mini Warehouse - Terminal 5"
echo "Baseline + Nav2"
echo "=========================================="
echo "Results: $RUN_DIR"
echo

{
    echo "timestamp=$STAMP"
    echo "method=sca_baseline"
    echo "motion_backend=nav2"
    echo "world=mini_warehouse"
    echo "sca_commit=$(git -C "$HOME/SCA-AIFNav-Project/sca_aifnav/runtime_ws/src/sca_aifnav" rev-parse HEAD)"
    echo "aimapp_reproduction_commit=$(git -C "$HOME/SCA-AIFNav-Project/aimapp/reproduction" rev-parse HEAD)"
} > "$RUN_DIR/metadata.txt"

echo "Waiting for required ROS interfaces..."

for topic in \
    /odom \
    /scan \
    /camera_front/image_raw \
    /camera_left/image_raw \
    /camera_right/image_raw
do
    if timeout 10 ros2 topic type "$topic" >/dev/null 2>&1; then
        echo "OK: $topic"
    else
        echo "ERROR: missing topic $topic"
        exit 1
    fi
done

if timeout 10 ros2 action type /navigate_to_pose >/dev/null 2>&1; then
    echo "OK: /navigate_to_pose"
else
    echo "ERROR: Nav2 /navigate_to_pose action is unavailable"
    exit 1
fi

echo
echo "Starting SCA-AIFNav baseline with Nav2..."
echo

ros2 run sca_aifnav_ros navigation_node \
    --ros-args \
    -p use_sim_time:=true \
    -p navigation_motion_backend:=nav2 \
    -p odom_topic:=/odom \
    -p agent_odom_topic:=/agent/odom \
    -p scan_topic:=/scan \
    -p camera_topic:=/camera_front/image_raw \
    -p left_camera_topic:=/camera_left/image_raw \
    -p right_camera_topic:=/camera_right/image_raw \
    -p cmd_vel_topic:=/cmd_vel \
    2>&1 | tee "$RUN_DIR/sca_agent.log"
