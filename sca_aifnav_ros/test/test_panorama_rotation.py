"""Tests for panorama rotation control mathematics."""

import math

import pytest

from sca_aifnav_ros.panorama_rotation import (
    ANGULAR_SPEED_GAIN,
    DEFAULT_MAX_ANGULAR_SPEED,
    GOAL_TOLERANCE_RAD,
    PanoramaRotationCommand,
    PanoramaRotationController,
)


def test_runtime_baseline_parameters():
    """Controller defaults should match panorama runtime behavior."""
    controller = PanoramaRotationController()

    assert (
        controller.max_angular_speed
        == pytest.approx(
            DEFAULT_MAX_ANGULAR_SPEED
        )
    )

    assert (
        controller.angular_speed_gain
        == pytest.approx(
            ANGULAR_SPEED_GAIN
        )
    )

    assert (
        controller.goal_tolerance_rad
        == pytest.approx(
            GOAL_TOLERANCE_RAD
        )
    )

    assert (
        DEFAULT_MAX_ANGULAR_SPEED
        == pytest.approx(0.2)
    )

    assert (
        ANGULAR_SPEED_GAIN
        == pytest.approx(0.6)
    )

    assert (
        GOAL_TOLERANCE_RAD
        == pytest.approx(0.1)
    )


def test_far_goal_uses_proportional_speed():
    """A moderate angular error should use proportional positive speed."""
    controller = PanoramaRotationController(
        max_angular_speed=1.0
    )

    result = controller.command(
        current_yaw_rad=0.0,
        goal_yaw_rad=0.2,
    )

    assert isinstance(
        result,
        PanoramaRotationCommand,
    )

    assert result.goal_reached is False

    assert (
        result.angular_speed_rad_s
        == pytest.approx(
            0.6 * 0.2
        )
    )


def test_angular_speed_is_capped():
    """Large angular error should respect the runtime speed limit."""
    controller = PanoramaRotationController()

    result = controller.command(
        current_yaw_rad=0.0,
        goal_yaw_rad=2.0,
    )

    assert result.goal_reached is False

    assert (
        result.angular_speed_rad_s
        == pytest.approx(
            DEFAULT_MAX_ANGULAR_SPEED
        )
    )


def test_exact_goal_has_zero_speed():
    """No rotation should be requested at the target orientation."""
    controller = PanoramaRotationController()

    result = controller.command(
        current_yaw_rad=1.0,
        goal_yaw_rad=1.0,
    )

    assert result.goal_reached is True

    assert (
        result.angular_error_rad
        == pytest.approx(0.0)
    )

    assert (
        result.angular_speed_rad_s
        == pytest.approx(0.0)
    )


def test_error_inside_tolerance_is_reached():
    """An angular error below the threshold should stop rotation."""
    controller = PanoramaRotationController()

    result = controller.command(
        current_yaw_rad=1.0,
        goal_yaw_rad=1.09,
    )

    assert result.goal_reached is True

    assert (
        result.angular_speed_rad_s
        == pytest.approx(0.0)
    )


def test_error_at_tolerance_is_reached():
    """The tolerance boundary should count as reaching the target."""
    controller = PanoramaRotationController()

    result = controller.command(
        current_yaw_rad=0.0,
        goal_yaw_rad=0.1,
    )

    assert result.goal_reached is True
    assert result.angular_speed_rad_s == 0.0


def test_error_above_tolerance_keeps_rotating():
    """An error just above tolerance should still request rotation."""
    controller = PanoramaRotationController(
        max_angular_speed=1.0
    )

    result = controller.command(
        current_yaw_rad=1.0,
        goal_yaw_rad=1.101,
    )

    assert result.goal_reached is False
    assert result.angular_speed_rad_s > 0.0


def test_zero_goal_wraps_to_two_pi_near_revolution_end():
    """A zero target near 2*pi should use the forward wrap convention."""
    controller = PanoramaRotationController()

    result = controller.command(
        current_yaw_rad=6.2,
        goal_yaw_rad=0.0,
    )

    assert (
        result.effective_goal_yaw_rad
        == pytest.approx(
            2.0 * math.pi
        )
    )

    assert (
        result.angular_error_rad
        == pytest.approx(
            2.0 * math.pi
            - 6.2
        )
    )

    assert result.goal_reached is True


def test_small_positive_goal_wraps_near_revolution_end():
    """A near-zero target should preserve the baseline clipping behavior."""
    controller = PanoramaRotationController()

    result = controller.command(
        current_yaw_rad=5.8,
        goal_yaw_rad=0.05,
    )

    assert (
        result.effective_goal_yaw_rad
        == pytest.approx(
            2.0 * math.pi
        )
    )

    assert result.goal_reached is False

    assert (
        result.angular_speed_rad_s
        == pytest.approx(
            min(
                0.6 * (
                    2.0 * math.pi
                    - 5.8
                ),
                DEFAULT_MAX_ANGULAR_SPEED,
            )
        )
    )


def test_zero_goal_does_not_wrap_when_current_yaw_is_low():
    """The special wrap rule should only apply near the revolution end."""
    controller = PanoramaRotationController(
        max_angular_speed=1.0
    )

    result = controller.command(
        current_yaw_rad=1.0,
        goal_yaw_rad=0.0,
    )

    assert (
        result.effective_goal_yaw_rad
        == pytest.approx(0.0)
    )

    assert (
        result.angular_error_rad
        == pytest.approx(1.0)
    )

    assert (
        result.angular_speed_rad_s
        == pytest.approx(0.6)
    )


def test_baseline_command_remains_positive_for_lower_goal():
    """Baseline panorama control should preserve positive speed semantics."""
    controller = PanoramaRotationController(
        max_angular_speed=1.0
    )

    result = controller.command(
        current_yaw_rad=2.0,
        goal_yaw_rad=1.0,
    )

    assert result.goal_reached is False

    assert (
        result.angular_speed_rad_s
        == pytest.approx(0.6)
    )

    assert result.angular_speed_rad_s > 0.0


@pytest.mark.parametrize(
    "field",
    [
        "current",
        "goal",
    ],
)
def test_non_finite_orientation_is_rejected(
    field,
):
    """Non-finite orientation inputs should not enter rotation control."""
    controller = PanoramaRotationController()

    current = 0.0
    goal = 1.0

    if field == "current":
        current = float("nan")
    else:
        goal = float("nan")

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        controller.command(
            current_yaw_rad=current,
            goal_yaw_rad=goal,
        )
