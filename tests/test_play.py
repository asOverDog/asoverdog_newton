import math

import pytest
import torch

import play as play_module
from control.config import default_asset_root, discover_robot_profiles
from control.state import RobotState
from play import PlayArgs, keyboard_command, parse_args, play


def test_play_exposes_discovered_profiles_and_default() -> None:
    source_assets = default_asset_root()
    profiles, default_robot = discover_robot_profiles(source_assets)
    parsed = parse_args([])
    assert parsed.asset_root == default_asset_root()
    assert parsed.robot is None
    for name in profiles:
        selected = parse_args(["--assets-root", str(source_assets), "--robot", name])
        assert selected.asset_root == source_assets
        assert selected.robot == name

    with pytest.raises(ValueError, match="unknown robot 'missing'") as error:
        play(PlayArgs(asset_root=source_assets, robot="missing"))
    assert str(source_assets) in str(error.value)
    assert default_robot in profiles


def test_play_selects_a_dated_policy_set() -> None:
    parsed = parse_args(["--policy", "2026-08-17", "--robot", "spherical"])

    assert parsed.policy == "2026-08-17"
    assert parsed.robot == "spherical"


def test_headless_requires_a_positive_step_count() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--headless"])


class _Keys:
    def __init__(self, pressed: set[str]) -> None:
        self.pressed = pressed

    def is_key_down(self, key: str) -> bool:
        return key in self.pressed


def test_keyboard_command_maps_translation_and_yaw() -> None:
    assert keyboard_command(_Keys({"i", "j", "u"})) == (1.0, 0.5, 1.0)


def test_heading_hold_corrects_drift_and_wraps_at_pi() -> None:
    heading = play_module.HeadingHold(policy_dt=0.02)
    heading.reset(3.1)

    assert heading.command((0.5, 0.0, 0.0), current_yaw=-3.1) == pytest.approx((0.5, 0.0, -0.08318530717958605))


def test_heading_hold_integrates_requested_turn_then_holds_the_new_heading() -> None:
    heading = play_module.HeadingHold(policy_dt=0.02)
    heading.reset(0.0)

    assert heading.command((0.5, 0.0, 1.0), current_yaw=0.0) == pytest.approx((0.5, 0.0, 1.0))
    assert heading.target_yaw == pytest.approx(0.02)
    assert heading.command((0.5, 0.0, 0.0), current_yaw=0.03) == pytest.approx((0.5, 0.0, -0.01))


def test_heading_hold_outputs_zero_and_reanchors_when_user_stops() -> None:
    heading = play_module.HeadingHold(policy_dt=0.02)
    heading.reset(0.0)

    assert heading.command((0.0, 0.0, 0.0), current_yaw=-0.3) == (0.0, 0.0, 0.0)
    assert heading.target_yaw == pytest.approx(-0.3)


def test_heading_hold_reset_and_output_limit() -> None:
    heading = play_module.HeadingHold(policy_dt=0.02)
    heading.reset(0.0)
    assert heading.command((0.5, 0.0, 0.0), current_yaw=-3.0)[2] == pytest.approx(2.0)

    heading.reset(-1.25)
    assert heading.target_yaw == pytest.approx(-1.25)
    assert heading.command((0.0, 0.0, 0.0), current_yaw=-1.25)[2] == pytest.approx(0.0)


class _RecordingController:
    device = torch.device("cpu")

    def __init__(self, view: RobotState) -> None:
        self.view = view
        self.current_action = torch.zeros(1, 12)
        self.commands: list[torch.Tensor] = []

    def step(self, command: torch.Tensor) -> RobotState:
        self.commands.append(command.clone())
        return self.view


def test_headless_loop_applies_heading_hold_to_policy_command() -> None:
    yaw = 0.25
    view = RobotState(
        base_pose=torch.tensor([[0.0, 0.0, 0.34, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]]),
        base_velocity=torch.zeros(1, 6),
        motor_position=torch.zeros(1, 12),
        motor_velocity=torch.zeros(1, 12),
        foot_pose=torch.zeros(1, 4, 7),
        foot_velocity=torch.zeros(1, 4, 6),
        loop_position_error=torch.zeros(1, 4),
        applied_motor_torque=torch.zeros(1, 12),
    )
    controller = _RecordingController(view)
    heading = play_module.HeadingHold(policy_dt=0.02)
    heading.reset(0.0)
    metrics = play_module._Metrics(initial_xy=torch.zeros(2))

    play_module._run_headless(
        controller,
        PlayArgs(headless=True, steps=1, vx=0.5),
        metrics,
        heading,
        view,
    )

    assert torch.allclose(controller.commands[0], torch.tensor([[0.5, 0.0, -0.25]]))
