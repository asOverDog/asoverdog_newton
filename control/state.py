"""Control-facing state and simulation contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RobotState:
    """Diagnostic tensors gathered from the current simulation state."""

    base_pose: torch.Tensor
    base_velocity: torch.Tensor
    motor_position: torch.Tensor
    motor_velocity: torch.Tensor
    foot_pose: torch.Tensor
    foot_velocity: torch.Tensor
    loop_position_error: torch.Tensor
    applied_motor_torque: torch.Tensor
