"""Tests for ROS 2 physical orientation adaptation."""

import math

from geometry_msgs.msg import Quaternion
import pytest

from sca_aifnav_ros.orientation_adapter import (
    OrientationAdapter,
)


def yaw_quaternion(
    yaw_rad,
):
    """Create a quaternion representing planar yaw."""
    quaternion = Quaternion()

    quaternion.z = math.sin(
        yaw_rad / 2.0
    )

    quaternion.w = math.cos(
        yaw_rad / 2.0
    )

    return quaternion


def test_identity_quaternion_has_zero_yaw():
    """Identity orientation should produce zero yaw."""
    adapter = OrientationAdapter()

    quaternion = Quaternion()
    quaternion.w = 1.0

    assert (
        adapter.yaw_from_quaternion(
            quaternion
        )
        == pytest.approx(0.0)
    )


def test_positive_ninety_degree_yaw():
    """Positive ninety degrees should remain in the first half turn."""
    adapter = OrientationAdapter()

    yaw = adapter.yaw_from_quaternion(
        yaw_quaternion(
            math.pi / 2.0
        )
    )

    assert yaw == pytest.approx(
        round(
            math.pi / 2.0,
            4,
        )
    )


def test_positive_pi_yaw():
    """A half turn should produce approximately pi radians."""
    adapter = OrientationAdapter()

    yaw = adapter.yaw_from_quaternion(
        yaw_quaternion(
            math.pi
        )
    )

    assert yaw == pytest.approx(
        round(
            math.pi,
            4,
        )
    )


def test_negative_ninety_degree_yaw_wraps_positive():
    """Negative yaw should shift into the positive angular interval."""
    adapter = OrientationAdapter()

    yaw = adapter.yaw_from_quaternion(
        yaw_quaternion(
            -math.pi / 2.0
        )
    )

    expected = (
        round(
            -math.pi / 2.0,
            4,
        )
        + 2.0
        * math.pi
    )

    assert yaw == pytest.approx(
        expected
    )

    assert (
        0.0
        <= yaw
        < 2.0 * math.pi
    )


def test_rounding_occurs_before_negative_angle_shift():
    """Baseline rounding order should be preserved for negative yaw."""
    adapter = OrientationAdapter()

    yaw = adapter.yaw_from_quaternion(
        yaw_quaternion(
            -0.123456
        )
    )

    expected = (
        round(
            -0.123456,
            4,
        )
        + 2.0
        * math.pi
    )

    assert yaw == pytest.approx(
        expected
    )


def test_non_unit_finite_quaternion_is_not_normalized():
    """Finite quaternion values should retain direct baseline semantics."""
    adapter = OrientationAdapter()

    quaternion = Quaternion()
    quaternion.z = 1.0
    quaternion.w = 1.0

    yaw = adapter.yaw_from_quaternion(
        quaternion
    )

    expected = round(
        math.atan2(
            2.0,
            -1.0,
        ),
        4,
    )

    assert yaw == pytest.approx(
        expected
    )


def test_zero_quaternion_preserves_direct_formula_behavior():
    """A finite zero quaternion should follow the direct yaw formula."""
    adapter = OrientationAdapter()

    quaternion = Quaternion()

    assert (
        adapter.yaw_from_quaternion(
            quaternion
        )
        == pytest.approx(0.0)
    )


def test_non_quaternion_input_is_rejected():
    """Unrelated objects should not be accepted as orientation."""
    adapter = OrientationAdapter()

    with pytest.raises(
        TypeError,
        match=(
            "quaternion must be "
            "geometry_msgs.msg.Quaternion"
        ),
    ):
        adapter.yaw_from_quaternion(
            object()
        )


@pytest.mark.parametrize(
    "field",
    [
        "x",
        "y",
        "z",
        "w",
    ],
)
def test_non_finite_component_is_rejected(
    field,
):
    """Non-finite quaternion components should be rejected."""
    adapter = OrientationAdapter()

    quaternion = Quaternion()
    quaternion.w = 1.0

    setattr(
        quaternion,
        field,
        math.nan,
    )

    with pytest.raises(
        ValueError,
        match=(
            "quaternion components must be finite"
        ),
    ):
        adapter.yaw_from_quaternion(
            quaternion
        )
