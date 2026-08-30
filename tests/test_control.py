import json

import torch

from control.actuator import Actuator
from control.config import RobotProfile, default_asset_root, discover_robot_profiles, get_robot_profile
from control.controller import Controller
from control.policy import ObservationHistory, OnnxPolicy, ZeroActionPolicy
from control.state import RobotState

PROFILES, DEFAULT_PROFILE = discover_robot_profiles(default_asset_root())


def _view(profile: RobotProfile) -> RobotState:
    action_dim = profile.action_dim
    foot_count = len(profile.model.foot_bodies)
    loop_count = len(profile.model.loop_joints)
    return RobotState(
        base_pose=torch.tensor([[0.0, 0.0, profile.reset_base_height, 0.0, 0.0, 0.0, 1.0]]),
        base_velocity=torch.zeros(1, 6),
        motor_position=torch.tensor(profile.actuator.home_position).reshape(1, action_dim),
        motor_velocity=torch.zeros(1, action_dim),
        foot_pose=torch.zeros(1, foot_count, 7),
        foot_velocity=torch.zeros(1, foot_count, 6),
        loop_position_error=torch.zeros(1, loop_count),
        applied_motor_torque=torch.zeros(1, action_dim),
    )


def test_policy_manifest_owns_control_timing() -> None:
    root = default_asset_root()
    robots = json.loads((root / "robots" / "robots.json").read_text())
    policies = json.loads((root / "policies" / "policies.json").read_text())

    assert "policy_decimation" not in robots["physics"]
    assert policies["default"] == "2026-08-17"
    assert set(policies["sets"]) == {"2026-08-17"}
    assert all(policy["control"]["decimation"] == 8 for policy in policies["sets"].values())


def test_shared_actuator_network_runs_for_every_robot() -> None:
    bennett = get_robot_profile("bennett", PROFILES)
    planar = get_robot_profile("planar", PROFILES)
    spherical = get_robot_profile("spherical", PROFILES)

    for profile in (bennett, planar, spherical):
        actuator = Actuator(profile, "cpu")
        torque = actuator.compute(
            torch.tensor(profile.actuator.home_position).reshape(1, -1),
            torch.tensor(profile.actuator.home_position).reshape(1, -1),
            torch.zeros(1, profile.action_dim),
        )
        assert torque.shape == (1, profile.action_dim)
        assert torch.isfinite(torque).all()


def test_all_external_policies_produce_finite_actions_and_zero_diagnostics() -> None:
    for profile in PROFILES.values():
        observation = torch.zeros(1, profile.observation.input_width)
        action = OnnxPolicy(profile.policy.path, profile)(observation)
        zero = ZeroActionPolicy("cpu", profile.observation.input_width, profile.action_dim)(observation)

        assert action.shape == (1, profile.action_dim)
        assert torch.isfinite(action).all()
        assert torch.equal(zero, torch.zeros_like(action))


def test_observation_history_shifts_and_resets() -> None:
    profile = get_robot_profile(DEFAULT_PROFILE, PROFILES)
    history = ObservationHistory(profile, "cpu")
    action = torch.arange(profile.action_dim, dtype=torch.float32).reshape(1, -1)
    action_history = torch.stack((action, action + 1.0), dim=1)
    view = _view(profile)
    command = torch.tensor([[0.2, -0.5, 0.75]])

    first = history.append(view, command, action_history).clone()
    second = history.append(view, command, action_history)
    width = profile.observation.frame_width

    assert torch.equal(second[:, -2 * width : -width], first[:, -width:])
    assert torch.equal(second[:, -action_history.numel() :], action_history.reshape(1, -1))
    history.reset()
    assert torch.count_nonzero(history.flattened) == 0


def test_observation_uses_manifest_standstill_clock() -> None:
    profile = get_robot_profile(DEFAULT_PROFILE, PROFILES)
    history = ObservationHistory(profile, "cpu")
    action_history = torch.zeros(1, profile.observation.action_history_length, profile.action_dim)

    observation = history.append(_view(profile), torch.zeros(1, 3), action_history)

    frame = observation[:, -profile.observation.frame_width :]
    assert torch.equal(frame[:, :4], torch.full((1, 4), 0.5))


class _ResetOnlySimulation:
    num_envs = 1
    device = torch.device("cpu")

    def __init__(self, profile: RobotProfile) -> None:
        self.profile = profile

    def reset(self, mask: torch.Tensor) -> torch.Tensor:
        return mask.clone()

    def view(self) -> RobotState:
        return _view(self.profile)


def test_controller_reset_clears_policy_state() -> None:
    profile = get_robot_profile(DEFAULT_PROFILE, PROFILES)
    controller = Controller(
        _ResetOnlySimulation(profile),
        profile,
        ZeroActionPolicy("cpu", profile.observation.input_width, profile.action_dim),
        Actuator(profile, "cpu"),
    )
    controller.action_history.fill_(1.0)
    controller.history.frames.fill_(3.0)

    controller.reset()

    assert torch.count_nonzero(controller.action_history) == 0
    assert torch.count_nonzero(controller.history.flattened) == 0
