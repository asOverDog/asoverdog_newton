"""Control-facing state and simulation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class RobotState:
    """Policy-facing tensors gathered from the current simulation state."""

    base_pose: torch.Tensor
    base_velocity: torch.Tensor
    motor_position: torch.Tensor
    motor_velocity: torch.Tensor
    foot_pose: torch.Tensor
    foot_velocity: torch.Tensor
    loop_position_error: torch.Tensor
    applied_motor_torque: torch.Tensor


class Simulation(Protocol):
    """Minimal simulator surface consumed by the policy controller."""

    num_envs: int
    device: torch.device

    def view(self) -> RobotState: ...

    def reset(self, mask: torch.Tensor) -> torch.Tensor: ...

    def step_motor_torques(
        self,
        torque: torch.Tensor,
        *,
        validate_finite: bool = True,
    ) -> RobotState: ...
