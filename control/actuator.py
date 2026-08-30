"""Robstride actuator network used by all three policies."""

from __future__ import annotations

import warnings

import torch

from control.config import RobotProfile, verify_actuator_asset


class Actuator:
    history_length = 6

    def __init__(self, profile: RobotProfile, device: str | torch.device) -> None:
        self.path = verify_actuator_asset(profile)
        self.sha256 = profile.actuator_sha256
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.num_joints = profile.action_dim
        self.effort_limit = profile.actuator.effort_limit
        self.velocity_limit = profile.actuator.velocity_limit
        self.saturation_effort = profile.actuator.saturation_effort
        # The supplied actuator asset is TorchScript; torch.export cannot load this format.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`torch.jit.load` is deprecated", category=DeprecationWarning)
            self.network = torch.jit.load(str(self.path), map_location=self.device).eval()
        history_shape = (1, self.history_length, self.num_joints)
        self.position_error_history = torch.zeros(history_shape, dtype=torch.float32, device=self.device)
        self.velocity_history = torch.zeros_like(self.position_error_history)

    def _validate(self, name: str, value: torch.Tensor, *, finite: bool) -> None:
        expected = (1, self.num_joints)
        if not isinstance(value, torch.Tensor) or value.shape != expected or value.dtype != torch.float32:
            raise ValueError(f"{name} must be a float32 Torch tensor with shape {expected}")
        if value.device != self.device:
            raise ValueError(f"{name} must be on {self.device}; got {value.device}")
        if finite and not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} must be finite")

    @torch.inference_mode()
    def compute(
        self,
        target_position: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        *,
        validate: bool = True,
    ) -> torch.Tensor:
        self._validate("target_position", target_position, finite=validate)
        self._validate("joint_position", joint_position, finite=validate)
        self._validate("joint_velocity", joint_velocity, finite=validate)
        self.position_error_history = torch.roll(self.position_error_history, shifts=1, dims=1)
        self.velocity_history = torch.roll(self.velocity_history, shifts=1, dims=1)
        self.position_error_history[:, 0] = target_position - joint_position
        self.velocity_history[:, 0] = joint_velocity
        network_input = (
            torch.cat((self.position_error_history * -1.0, self.velocity_history), dim=1)
            .transpose(1, 2)
            .reshape(self.num_joints, 2 * self.history_length)
        )
        raw_torque = self.network(network_input).reshape(1, self.num_joints)
        limited_velocity = joint_velocity.clamp(-2.0 * self.velocity_limit, 2.0 * self.velocity_limit)
        upper = (self.saturation_effort * (1.0 - limited_velocity / self.velocity_limit)).clamp(max=self.effort_limit)
        lower = (self.saturation_effort * (-1.0 - limited_velocity / self.velocity_limit)).clamp(min=-self.effort_limit)
        torque = torch.clamp(raw_torque, min=lower, max=upper)
        if validate and not torch.isfinite(torque).all():
            raise FloatingPointError("actuator network torque must be finite")
        return torque

    def reset(self) -> None:
        self.position_error_history.zero_()
        self.velocity_history.zero_()
