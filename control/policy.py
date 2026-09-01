"""Device-resident gait policy inference."""

from __future__ import annotations

import hashlib
from pathlib import Path

import warp as wp
from warp_nn.runtime import OnnxRuntime

from control.config import RobotProfile, verify_policy_asset


class OnnxPolicy:
    """Run an exported gait policy without leaving the Warp device."""

    mode = "onnx"

    def __init__(self, path: Path, profile: RobotProfile, device: str | wp.Device) -> None:
        if not isinstance(profile, RobotProfile):
            raise TypeError("profile must be a RobotProfile")
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"ONNX policy does not exist: {resolved}")
        if resolved == profile.policy.path.resolve():
            verify_policy_asset(profile)

        self.profile = profile
        self.device = wp.get_device(device)
        self.action_width = profile.policy.output.width
        self.path = resolved
        self.sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
        self._runtime = OnnxRuntime(str(resolved), device=self.device)
        expected_input = profile.policy.input.name
        expected_output = profile.policy.output.name
        if self._runtime.input_names != [expected_input]:
            raise ValueError(f"{profile.name} ONNX input must be {expected_input!r}; got {self._runtime.input_names}")
        if self._runtime.output_names != [expected_output]:
            raise ValueError(
                f"{profile.name} ONNX output must be {expected_output!r}; got {self._runtime.output_names}"
            )
        self._input_name = expected_input
        self._output_name = expected_output

    def __call__(self, observation: wp.array) -> wp.array:
        _validate_observation(observation, self.profile.observation.input_width, self.device)
        output = self._runtime({self._input_name: observation})[self._output_name]
        if output.shape != (1, self.action_width):
            raise ValueError(f"policy output must have shape (1, {self.action_width}); got {output.shape}")
        return output


class ZeroActionPolicy:
    """Return a fixed device-resident zero action for diagnostics."""

    mode = "zero_action"
    path = None
    sha256 = None

    def __init__(self, device: str | wp.Device, input_width: int, action_width: int) -> None:
        if not isinstance(input_width, int) or isinstance(input_width, bool) or input_width <= 0:
            raise ValueError("input_width must be a positive integer")
        if not isinstance(action_width, int) or isinstance(action_width, bool) or action_width <= 0:
            raise ValueError("action_width must be a positive integer")
        self.device = wp.get_device(device)
        self.input_width = input_width
        self.action_width = action_width
        self._action = wp.zeros((1, action_width), dtype=wp.float32, device=self.device)

    def __call__(self, observation: wp.array) -> wp.array:
        _validate_observation(observation, self.input_width, self.device)
        return self._action


def _validate_observation(observation: wp.array, input_width: int, device: wp.Device) -> None:
    if not hasattr(observation, "shape") or observation.shape != (1, input_width):
        shape = getattr(observation, "shape", None)
        raise ValueError(f"observation shape must be (1, {input_width}); got {shape}")
    if observation.dtype != wp.float32:
        raise TypeError(f"observation must be float32; got {observation.dtype}")
    if observation.device != device:
        raise ValueError(f"observation must be on {device}; got {observation.device}")
