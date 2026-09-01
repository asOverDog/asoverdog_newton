import json

import numpy as np
import warp as wp

from control.actuator import Actuator
from control.config import default_asset_root, discover_robot_profiles, get_robot_profile
from control.policy import OnnxPolicy, ZeroActionPolicy

PROFILES, DEFAULT_PROFILE = discover_robot_profiles(default_asset_root())


def test_policy_manifest_owns_control_timing() -> None:
    root = default_asset_root()
    robots = json.loads((root / "robots" / "robots.json").read_text())
    policies = json.loads((root / "policies" / "policies.json").read_text())

    assert "policy_decimation" not in robots["physics"]
    assert policies["default"] == "2026-08-17"
    assert set(policies["sets"]) == {"2026-08-17"}
    assert all(policy["control"]["decimation"] == 8 for policy in policies["sets"].values())


def test_control_assets_support_device_resident_inference() -> None:
    profile = get_robot_profile(DEFAULT_PROFILE, PROFILES)

    assert profile.actuator_path.suffix == ".npz"
    observation = wp.zeros((1, profile.observation.input_width), dtype=wp.float32, device="cpu")
    action = OnnxPolicy(profile.policy.path, profile, device="cpu")(observation)

    assert action.device == wp.get_device("cpu")
    assert action.shape == (1, profile.action_dim)


def test_shared_actuator_network_runs_for_every_robot() -> None:
    for profile in PROFILES.values():
        actuator = Actuator(profile, "cpu")
        network_input = wp.zeros((profile.action_dim, 2 * actuator.history_length), dtype=wp.float32, device="cpu")
        torque = actuator(network_input)

        assert torque.shape == (profile.action_dim, 1)
        assert np.isfinite(torque.numpy()).all()


def test_all_external_policies_produce_finite_actions_and_zero_diagnostics() -> None:
    for profile in PROFILES.values():
        observation = wp.zeros((1, profile.observation.input_width), dtype=wp.float32, device="cpu")
        action = OnnxPolicy(profile.policy.path, profile, device="cpu")(observation)
        zero = ZeroActionPolicy("cpu", profile.observation.input_width, profile.action_dim)(observation)

        assert action.shape == (1, profile.action_dim)
        assert np.isfinite(action.numpy()).all()
        assert np.count_nonzero(zero.numpy()) == 0
