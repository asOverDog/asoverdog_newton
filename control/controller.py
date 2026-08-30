"""One-policy-period control loop over a single Kamino world."""

from __future__ import annotations

from collections.abc import Callable

import torch

from control.actuator import Actuator
from control.config import RobotProfile
from control.policy import ObservationHistory
from control.state import RobotState, Simulation


class Controller:
    """Route one gait policy through its actuator network and physics substeps."""

    def __init__(
        self,
        simulation: Simulation,
        profile: RobotProfile,
        policy: Callable[[torch.Tensor], torch.Tensor],
        actuator: Actuator,
    ) -> None:
        if getattr(simulation, "num_envs", None) != 1:
            raise ValueError("control requires exactly one world")
        self.simulation = simulation
        self.profile = profile
        self.cfg = profile.config
        self.policy = policy
        self.actuator = actuator
        self.device = torch.device(getattr(simulation, "device"))
        self.home_motor_q = torch.tensor(profile.actuator.home_position, dtype=torch.float32, device=self.device)
        indices = tuple(profile.observation.policy_motor_order.index(name) for name in profile.model.motor_joints)
        self._sim_from_policy = torch.tensor(indices, dtype=torch.long, device=self.device)
        self.history = ObservationHistory(profile, self.device)
        self.action_history = torch.zeros(
            1,
            profile.observation.action_history_length,
            self.cfg.action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        self.last_target = self.home_motor_q.reshape(1, -1).clone()
        self.last_torque = torch.zeros(1, self.cfg.action_dim, dtype=torch.float32, device=self.device)
        self._runtime_invalid = torch.zeros((), dtype=torch.bool, device=self.device)

    @property
    def current_action(self) -> torch.Tensor:
        return self.action_history[:, 0]

    def _validate_view(self, view: RobotState, *, finite: bool = True) -> None:
        values = (
            (view.base_pose, (1, 7)),
            (view.base_velocity, (1, 6)),
            (view.motor_position, (1, self.profile.action_dim)),
            (view.motor_velocity, (1, self.profile.action_dim)),
            (view.foot_pose, (1, len(self.profile.model.foot_bodies), 7)),
            (view.foot_velocity, (1, len(self.profile.model.foot_bodies), 6)),
            (view.loop_position_error, (1, len(self.profile.model.loop_joints))),
            (view.applied_motor_torque, (1, self.profile.action_dim)),
        )
        for value, shape in values:
            if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32:
                raise ValueError(f"simulation view fields must be float32 tensors; expected {shape}")
            if value.device != self.device:
                raise ValueError(f"simulation view must be on {self.device}; got {value.device}")
        if finite and not torch.cat([value.reshape(-1) for value, _ in values]).isfinite().all():
            raise FloatingPointError("simulation view must be finite")

    def step(self, command: torch.Tensor) -> RobotState:
        if command.shape != (1, 3) or command.dtype != torch.float32 or command.device != self.device:
            raise ValueError("command must be a device-local float32[1,3] tensor")
        if not torch.isfinite(command).all():
            raise FloatingPointError("command must be finite")
        initial_view = self.simulation.view()
        self._validate_view(initial_view, finite=False)
        observation = self.history.append(
            initial_view,
            command,
            self.action_history,
        )
        action = self.policy(observation)
        expected = (1, self.profile.action_dim)
        if action.shape != expected or action.dtype != torch.float32 or action.device != self.device:
            raise ValueError(f"policy action must be a device-local float32 tensor with shape {expected}")
        if not torch.isfinite(action).all():
            raise FloatingPointError("policy action must be finite")
        if self.profile.observation.action_history_length > 1:
            self.action_history[:, 1:].copy_(self.action_history[:, :-1].clone())
        self.action_history[:, 0].copy_(action).clamp_(-self.cfg.action_clip, self.cfg.action_clip)
        simulation_action = self.current_action.index_select(1, self._sim_from_policy)
        self.last_target.copy_(self.home_motor_q + self.cfg.actuator.action_scale * simulation_action)

        self._runtime_invalid.zero_()
        final_view = initial_view
        for _ in range(self.cfg.physics.policy_decimation):
            current_view = self.simulation.view()
            self._runtime_invalid.logical_or_(
                ~(torch.isfinite(current_view.motor_position).all() & torch.isfinite(current_view.motor_velocity).all())
            )
            torque = self.actuator.compute(
                self.last_target,
                torch.nan_to_num(current_view.motor_position),
                torch.nan_to_num(current_view.motor_velocity),
                validate=False,
            )
            self._runtime_invalid.logical_or_(~torch.isfinite(torque).all())
            self.last_torque.copy_(torch.nan_to_num(torque))
            final_view = self.simulation.step_motor_torques(self.last_torque, validate_finite=False)
        self._validate_view(final_view, finite=False)
        self._runtime_invalid.logical_or_(
            ~torch.cat(
                (
                    final_view.base_pose.reshape(-1),
                    final_view.base_velocity.reshape(-1),
                    final_view.motor_position.reshape(-1),
                    final_view.motor_velocity.reshape(-1),
                    final_view.foot_pose.reshape(-1),
                    final_view.foot_velocity.reshape(-1),
                    final_view.loop_position_error.reshape(-1),
                )
            )
            .isfinite()
            .all()
        )
        if bool(self._runtime_invalid):
            raise FloatingPointError("non-finite control value was sanitized before DVI")
        return final_view

    def reset(self) -> RobotState:
        mask = torch.ones(1, dtype=torch.bool, device=self.device)
        success = self.simulation.reset(mask)
        if not isinstance(success, torch.Tensor) or success.shape != (1,) or success.dtype != torch.bool:
            raise ValueError("reset success must be a bool[1] Torch tensor")
        if success.device != self.device or not torch.equal(success, mask):
            raise RuntimeError("Kamino control reset failed for world 0")
        view = self.simulation.view()
        self._validate_view(view)
        self.history.reset()
        self.action_history.zero_()
        self.last_target.copy_(self.home_motor_q)
        self.last_torque.zero_()
        self.actuator.reset()
        return view
