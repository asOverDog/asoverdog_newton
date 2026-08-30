"""Strict ONNX and zero-action policies for external gait profiles."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from control.config import RobotProfile, verify_policy_asset
from control.state import RobotState


def _quaternion_rotate_inverse_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    inverse_xyz = -quaternion[..., :3]
    twice_cross = 2.0 * torch.linalg.cross(inverse_xyz, vector, dim=-1)
    return vector + quaternion[..., 3:4] * twice_cross + torch.linalg.cross(inverse_xyz, twice_cross, dim=-1)


class ObservationHistory:
    """Oldest-to-newest history matching the exported gait policies."""

    def __init__(self, profile: RobotProfile, device: str | torch.device) -> None:
        if not isinstance(profile, RobotProfile):
            raise TypeError("profile must be a RobotProfile")
        self.profile = profile
        self.config = profile.observation
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.frames = torch.zeros(
            1, self.config.history_length, self.config.frame_width, dtype=torch.float32, device=self.device
        )
        self._phase = torch.tensor([self.config.clock.phase_offsets], dtype=torch.float32, device=self.device)
        self._initial_phase = self._phase.clone()
        self._gravity_world = torch.tensor([self.config.gravity_direction], dtype=torch.float32, device=self.device)
        self._standstill_threshold = torch.tensor(
            [self.config.clock.command_deadband], dtype=torch.float32, device=self.device
        )
        self._standstill_clock = torch.tensor(
            [self.config.clock.standstill_values], dtype=torch.float32, device=self.device
        )
        self._home_motor_q = torch.tensor(profile.actuator.home_position, dtype=torch.float32, device=self.device)
        indices = tuple(profile.model.motor_joints.index(name) for name in self.config.policy_motor_order)
        self._policy_from_sim = torch.tensor(indices, dtype=torch.long, device=self.device)

    @property
    def flattened(self) -> torch.Tensor:
        return self.frames.reshape(1, self.config.input_width)

    def reset(self) -> None:
        self.frames.zero_()
        self._phase.copy_(self._initial_phase)

    def _validate_tensor(self, name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
        if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32:
            raise ValueError(f"{name} must be a float32 Torch tensor with shape {shape}")
        if value.device != self.device:
            raise ValueError(f"{name} must be on {self.device}; got {value.device}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} must be finite")

    def _clock_and_command(self, command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        standing = (command.abs() < self._standstill_threshold).all(dim=1)
        effective_command = torch.where(standing.unsqueeze(1), torch.zeros_like(command), command)
        phase_step = self.profile.config.physics.policy_dt * self.config.clock.frequency_hz * 2.0
        if self.config.clock.advance_before_observation:
            self._phase.add_(phase_step)
        clock = torch.sin(math.pi * self._phase)
        clock = torch.where(standing.unsqueeze(1), self._standstill_clock, clock)
        if not self.config.clock.advance_before_observation:
            self._phase.add_(phase_step)
        self._phase.remainder_(2.0)
        return clock, effective_command

    def append(
        self,
        view: RobotState,
        command: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        for name, value, shape in (
            ("base_pose", view.base_pose, (1, 7)),
            ("base_velocity", view.base_velocity, (1, 6)),
            ("motor_position", view.motor_position, (1, self.profile.action_dim)),
            ("motor_velocity", view.motor_velocity, (1, self.profile.action_dim)),
            ("command", command, (1, 3)),
            (
                "action_history",
                action_history,
                (1, self.config.action_history_length, self.profile.action_dim),
            ),
        ):
            self._validate_tensor(name, value, shape)
        clock, effective_command = self._clock_and_command(command)
        quaternion = view.base_pose[:, 3:7]
        values = {
            "gait_clock": clock,
            "projected_gravity": _quaternion_rotate_inverse_xyzw(quaternion, self._gravity_world),
            "base_angular_velocity": _quaternion_rotate_inverse_xyzw(quaternion, view.base_velocity[:, 3:6]),
            "velocity_command": effective_command,
            "motor_position_error": (view.motor_position - self._home_motor_q).index_select(1, self._policy_from_sim),
            "motor_velocity": view.motor_velocity.index_select(1, self._policy_from_sim),
            "current_action": action_history[:, 0],
            "previous_action": action_history[:, 1],
        }
        current = torch.cat([values[field.name] * field.scale for field in self.config.fields], dim=1)
        if current.shape != (1, self.config.frame_width) or not torch.isfinite(current).all():
            raise FloatingPointError(f"invalid {self.profile.name} observation frame")
        self.frames[:, :-1].copy_(self.frames[:, 1:].clone())
        self.frames[:, -1].copy_(current)
        return self.flattened


class OnnxPolicy:
    """ONNX Runtime wrapper parameterized by one exported profile."""

    mode = "onnx"

    def __init__(self, path: Path, profile: RobotProfile) -> None:
        if not isinstance(profile, RobotProfile):
            raise TypeError("profile must be a RobotProfile")
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"ONNX policy does not exist: {resolved}")
        if resolved == profile.policy.path.resolve():
            verify_policy_asset(profile)
        self.profile = profile
        self.action_width = profile.policy.output.width
        self.path = resolved
        self.sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
        self._session = ort.InferenceSession(str(resolved), providers=["CPUExecutionProvider"])
        self._validate_graph_contract()

    @staticmethod
    def _node_signature(node: object) -> tuple[str, str, list[object]]:
        return str(getattr(node, "name", "")), str(getattr(node, "type", "")), list(getattr(node, "shape", []))

    def _validate_graph_contract(self) -> None:
        policy_input = self.profile.policy.input
        expected_input = (policy_input.name, policy_input.dtype, list(policy_input.shape))
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or self._node_signature(inputs[0]) != expected_input:
            found = [self._node_signature(node) for node in inputs]
            raise ValueError(f"{self.profile.name} ONNX input must be {expected_input}; got {found}")
        policy_output = self.profile.policy.output
        expected_output = (policy_output.name, policy_output.dtype, list(policy_output.shape))
        outputs = self._session.get_outputs()
        if len(outputs) != 1 or self._node_signature(outputs[0]) != expected_output:
            found = [self._node_signature(node) for node in outputs]
            raise ValueError(f"ONNX output must be {expected_output}; got {found}")

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        _validate_observation(observation, self.profile.observation.input_width)
        array = observation.detach().to("cpu").contiguous().numpy()
        output = self._session.run(
            [self.profile.policy.output.name],
            {self.profile.policy.input.name: array},
        )[0]
        if not isinstance(output, np.ndarray) or output.shape != (1, self.action_width):
            raise ValueError(f"ONNX action output must be a NumPy float32[1,{self.action_width}] array")
        if output.dtype != np.float32:
            raise TypeError(f"ONNX action output must be float32; got {output.dtype}")
        if not np.isfinite(output).all():
            raise FloatingPointError("ONNX action output must be finite")
        return torch.from_numpy(output.copy()).to(observation.device)


class ZeroActionPolicy:
    """Profile-width diagnostic policy returning device-local zeros."""

    mode = "zero_action"
    path = None
    sha256 = None

    def __init__(self, device: str | torch.device, input_width: int, action_width: int) -> None:
        if not isinstance(input_width, int) or isinstance(input_width, bool) or input_width <= 0:
            raise ValueError("input_width must be a positive integer")
        if not isinstance(action_width, int) or isinstance(action_width, bool) or action_width <= 0:
            raise ValueError("action_width must be a positive integer")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.input_width = input_width
        self.action_width = action_width
        self._action = torch.zeros(1, self.action_width, dtype=torch.float32, device=self.device)

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        _validate_observation(observation, self.input_width, device=self.device)
        return self._action


def _validate_observation(
    observation: torch.Tensor,
    input_width: int,
    *,
    device: torch.device | None = None,
) -> None:
    if not isinstance(observation, torch.Tensor):
        raise TypeError("observation must be a Torch tensor")
    if observation.shape != (1, input_width):
        raise ValueError(f"observation shape must be (1, {input_width}); got {tuple(observation.shape)}")
    if observation.dtype != torch.float32:
        raise TypeError(f"observation must be float32; got {observation.dtype}")
    if device is not None and observation.device != device:
        raise ValueError(f"observation must be on {device}; got {observation.device}")
    if not torch.isfinite(observation).all():
        raise FloatingPointError("observation must be finite")
