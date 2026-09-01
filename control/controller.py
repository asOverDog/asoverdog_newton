"""Device-resident policy-period control for one Kamino world."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import warp as wp

from control.actuator import Actuator
from control.config import RobotProfile
from control.state import RobotState

ACTION_COUNT = 12
OBSERVATION_FRAME_SIZE = 61
OBSERVATION_HISTORY = 25
ACTUATOR_HISTORY = 6


@wp.kernel
def _build_observation(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    base_body: int,
    policy_coord_indices: wp.array(dtype=int),
    policy_dof_indices: wp.array(dtype=int),
    policy_home: wp.array(dtype=float),
    command: wp.array(dtype=wp.vec3),
    phase: wp.array(dtype=wp.vec4),
    phase_step: float,
    command_deadband: wp.vec3,
    standstill_clock: wp.vec4,
    gravity_world: wp.vec3,
    action_history: wp.array2d(dtype=float),
    observation: wp.array2d(dtype=float),
):
    for i in range((OBSERVATION_HISTORY - 1) * OBSERVATION_FRAME_SIZE):
        observation[0, i] = observation[0, i + OBSERVATION_FRAME_SIZE]

    cmd = command[0]
    standing = (
        wp.abs(cmd[0]) < command_deadband[0]
        and wp.abs(cmd[1]) < command_deadband[1]
        and wp.abs(cmd[2]) < command_deadband[2]
    )
    effective_cmd = cmd
    if standing:
        effective_cmd = wp.vec3(0.0, 0.0, 0.0)

    gait_phase = phase[0] + wp.vec4(phase_step, phase_step, phase_step, phase_step)
    for i in range(4):
        if gait_phase[i] >= 2.0:
            gait_phase[i] = gait_phase[i] - 2.0
    phase[0] = gait_phase

    base = body_q[base_body]
    rotation = wp.transform_get_rotation(base)
    angular_velocity = wp.quat_rotate_inv(rotation, wp.spatial_bottom(body_qd[base_body]))
    gravity = wp.quat_rotate_inv(rotation, gravity_world)
    offset = (OBSERVATION_HISTORY - 1) * OBSERVATION_FRAME_SIZE

    for i in range(4):
        clock = wp.sin(wp.pi * gait_phase[i])
        if standing:
            clock = standstill_clock[i]
        observation[0, offset + i] = clock

    observation[0, offset + 4] = gravity[0]
    observation[0, offset + 5] = gravity[1]
    observation[0, offset + 6] = gravity[2]
    observation[0, offset + 7] = 2.0 * angular_velocity[0]
    observation[0, offset + 8] = 2.0 * angular_velocity[1]
    observation[0, offset + 9] = 2.0 * angular_velocity[2]
    observation[0, offset + 10] = 2.0 * effective_cmd[0]
    observation[0, offset + 11] = 2.0 * effective_cmd[1]
    observation[0, offset + 12] = 2.0 * effective_cmd[2]

    for i in range(ACTION_COUNT):
        observation[0, offset + 13 + i] = joint_q[policy_coord_indices[i]] - policy_home[i]
        observation[0, offset + 25 + i] = 0.05 * joint_qd[policy_dof_indices[i]]
        observation[0, offset + 37 + i] = action_history[0, i]
        observation[0, offset + 49 + i] = action_history[1, i]


@wp.kernel
def _update_action(
    policy_action: wp.array2d(dtype=float),
    action_clip: float,
    action_history: wp.array2d(dtype=float),
):
    i = wp.tid()
    action_history[1, i] = action_history[0, i]
    action_history[0, i] = wp.clamp(policy_action[0, i], -action_clip, action_clip)


@wp.kernel
def _build_actuator_input(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    motor_coord_indices: wp.array(dtype=int),
    motor_dof_indices: wp.array(dtype=int),
    sim_from_policy: wp.array(dtype=int),
    home_position: wp.array(dtype=float),
    action_scale: float,
    action_history: wp.array2d(dtype=float),
    position_error_history: wp.array2d(dtype=float),
    velocity_history: wp.array2d(dtype=float),
    actuator_input: wp.array2d(dtype=float),
):
    motor = wp.tid()
    for history in range(ACTUATOR_HISTORY - 1, 0, -1):
        position_error_history[history, motor] = position_error_history[history - 1, motor]
        velocity_history[history, motor] = velocity_history[history - 1, motor]

    target = home_position[motor] + action_scale * action_history[0, sim_from_policy[motor]]
    position_error_history[0, motor] = target - joint_q[motor_coord_indices[motor]]
    velocity_history[0, motor] = joint_qd[motor_dof_indices[motor]]
    for history in range(ACTUATOR_HISTORY):
        actuator_input[motor, history] = -position_error_history[history, motor]
        actuator_input[motor, ACTUATOR_HISTORY + history] = velocity_history[history, motor]


@wp.kernel
def _apply_motor_torque(
    raw_torque: wp.array2d(dtype=float),
    joint_qd: wp.array(dtype=float),
    motor_dof_indices: wp.array(dtype=int),
    effort_limit: float,
    velocity_limit: float,
    saturation_effort: float,
    joint_force: wp.array(dtype=float),
    last_torque: wp.array(dtype=float),
):
    motor = wp.tid()
    dof = motor_dof_indices[motor]
    velocity = wp.clamp(joint_qd[dof], -2.0 * velocity_limit, 2.0 * velocity_limit)
    upper = wp.min(saturation_effort * (1.0 - velocity / velocity_limit), effort_limit)
    lower = wp.max(saturation_effort * (-1.0 - velocity / velocity_limit), -effort_limit)
    torque = wp.clamp(raw_torque[motor, 0], lower, upper)
    joint_force[dof] = torque
    last_torque[motor] = torque


class Controller:
    """Route one gait policy through its actuator network and eight physics steps."""

    def __init__(
        self,
        simulation,
        profile: RobotProfile,
        policy: Callable[[wp.array], wp.array],
        actuator: Actuator,
        *,
        use_cuda_graph: bool = True,
    ) -> None:
        if getattr(simulation, "num_envs", None) != 1:
            raise ValueError("control requires exactly one world")
        self._validate_policy_contract(profile)
        self.simulation = simulation
        self.profile = profile
        self.cfg = profile.config
        self.policy = policy
        self.actuator = actuator
        self.device = wp.get_device(str(simulation.device))

        motor_coords = simulation.layout.motor_coord_indices[0].cpu().tolist()
        motor_dofs = simulation.layout.motor_dof_indices[0].cpu().tolist()
        policy_coords = [
            motor_coords[profile.model.motor_joints.index(name)] for name in profile.observation.policy_motor_order
        ]
        policy_dofs = [
            motor_dofs[profile.model.motor_joints.index(name)] for name in profile.observation.policy_motor_order
        ]
        sim_from_policy = [profile.observation.policy_motor_order.index(name) for name in profile.model.motor_joints]

        self._base_body = int(simulation.layout.base_body_indices[0].item())
        self._motor_coord_indices = wp.array(motor_coords, dtype=wp.int32, device=self.device)
        self._motor_dof_indices = wp.array(motor_dofs, dtype=wp.int32, device=self.device)
        self._policy_coord_indices = wp.array(policy_coords, dtype=wp.int32, device=self.device)
        self._policy_dof_indices = wp.array(policy_dofs, dtype=wp.int32, device=self.device)
        self._sim_from_policy = wp.array(sim_from_policy, dtype=wp.int32, device=self.device)
        self._home_position = wp.array(profile.actuator.home_position, dtype=wp.float32, device=self.device)
        policy_home = [
            profile.actuator.home_position[profile.model.motor_joints.index(name)]
            for name in profile.observation.policy_motor_order
        ]
        self._policy_home = wp.array(policy_home, dtype=wp.float32, device=self.device)

        clock = profile.observation.clock
        self._phase_step = self.cfg.physics.policy_dt * clock.frequency_hz * 2.0
        self._initial_phase = wp.vec4(*clock.phase_offsets)
        self._phase = wp.array([self._initial_phase], dtype=wp.vec4, device=self.device)
        self._command_deadband = wp.vec3(*clock.command_deadband)
        self._standstill_clock = wp.vec4(*clock.standstill_values)
        self._gravity_world = wp.vec3(*profile.observation.gravity_direction)
        self._command = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self._action_history = wp.zeros((2, ACTION_COUNT), dtype=wp.float32, device=self.device)
        self._observation = wp.zeros(
            (1, OBSERVATION_HISTORY * OBSERVATION_FRAME_SIZE), dtype=wp.float32, device=self.device
        )
        self._position_error_history = wp.zeros((ACTUATOR_HISTORY, ACTION_COUNT), dtype=wp.float32, device=self.device)
        self._velocity_history = wp.zeros_like(self._position_error_history)
        self._actuator_input = wp.zeros((ACTION_COUNT, 2 * ACTUATOR_HISTORY), dtype=wp.float32, device=self.device)
        self._last_torque = wp.zeros(ACTION_COUNT, dtype=wp.float32, device=self.device)
        self._current_action_torch = wp.to_torch(self._action_history)[0].reshape(1, ACTION_COUNT)
        self._last_torque_torch = wp.to_torch(self._last_torque).reshape(1, ACTION_COUNT)

        self.graph = None
        self._warmup_networks()
        self.reset()
        if use_cuda_graph and self.device.is_cuda and wp.is_mempool_enabled(self.device):
            self._simulate_policy_period()
            self.reset()
            with wp.ScopedCapture(device=self.device) as capture:
                self._simulate_policy_period()
            self.graph = capture.graph
            self.reset()

    @staticmethod
    def _validate_policy_contract(profile: RobotProfile) -> None:
        fields = tuple((field.name, field.width, field.scale) for field in profile.observation.fields)
        expected = (
            ("gait_clock", 4, 1.0),
            ("projected_gravity", 3, 1.0),
            ("base_angular_velocity", 3, 2.0),
            ("velocity_command", 3, 2.0),
            ("motor_position_error", 12, 1.0),
            ("motor_velocity", 12, 0.05),
            ("current_action", 12, 1.0),
            ("previous_action", 12, 1.0),
        )
        if profile.action_dim != ACTION_COUNT or fields != expected:
            raise ValueError("device-resident controller requires the exported 61-value asOverDog observation contract")
        if profile.observation.history_length != OBSERVATION_HISTORY:
            raise ValueError(f"observation history must contain {OBSERVATION_HISTORY} frames")
        if profile.config.physics.policy_decimation != 8:
            raise ValueError("device-resident controller requires eight physics steps per policy period")
        if not profile.observation.clock.advance_before_observation:
            raise ValueError("gait phase must advance before constructing each observation")

    @property
    def cuda_graph_enabled(self) -> bool:
        return self.graph is not None

    @property
    def current_action(self) -> torch.Tensor:
        return self._current_action_torch

    @property
    def last_torque(self) -> torch.Tensor:
        return self._last_torque_torch

    def _warmup_networks(self) -> None:
        output = self.policy(self._observation)
        self.actuator(self._actuator_input)
        if output.shape != (1, ACTION_COUNT):
            raise ValueError(f"policy output must have shape (1, {ACTION_COUNT})")

    def _simulate_policy_period(self) -> None:
        state = self.simulation.current_state
        wp.launch(
            _build_observation,
            dim=1,
            inputs=[
                state.body_q,
                state.body_qd,
                state.joint_q,
                state.joint_qd,
                self._base_body,
                self._policy_coord_indices,
                self._policy_dof_indices,
                self._policy_home,
                self._command,
                self._phase,
                self._phase_step,
                self._command_deadband,
                self._standstill_clock,
                self._gravity_world,
                self._action_history,
                self._observation,
            ],
            device=self.device,
        )
        policy_action = self.policy(self._observation)
        wp.launch(
            _update_action,
            dim=ACTION_COUNT,
            inputs=[policy_action, self.cfg.action_clip, self._action_history],
            device=self.device,
        )

        for _ in range(self.cfg.physics.policy_decimation):
            state = self.simulation.current_state
            wp.launch(
                _build_actuator_input,
                dim=ACTION_COUNT,
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    self._motor_coord_indices,
                    self._motor_dof_indices,
                    self._sim_from_policy,
                    self._home_position,
                    self.cfg.actuator.action_scale,
                    self._action_history,
                    self._position_error_history,
                    self._velocity_history,
                    self._actuator_input,
                ],
                device=self.device,
            )
            raw_torque = self.actuator(self._actuator_input)
            self.simulation.control.joint_f.zero_()
            wp.launch(
                _apply_motor_torque,
                dim=ACTION_COUNT,
                inputs=[
                    raw_torque,
                    state.joint_qd,
                    self._motor_dof_indices,
                    self.cfg.actuator.effort_limit,
                    self.cfg.actuator.velocity_limit,
                    self.cfg.actuator.saturation_effort,
                    self.simulation.control.joint_f,
                    self._last_torque,
                ],
                device=self.device,
            )
            self.simulation.simulate_once()

    def step(self, command: tuple[float, float, float]) -> RobotState:
        if len(command) != 3 or not all(math.isfinite(value) for value in command):
            raise ValueError("command must contain three finite values")
        self._command.assign([wp.vec3(*command)])
        if self.graph is None:
            self._simulate_policy_period()
        else:
            wp.capture_launch(self.graph)
        return self.simulation.view(last_motor_torque=self.last_torque)

    def reset(self) -> RobotState:
        mask = torch.ones(1, dtype=torch.bool, device=self.simulation.device)
        success = self.simulation.reset(mask)
        if not torch.equal(success, mask):
            raise RuntimeError("Kamino control reset failed for world 0")
        self._command.zero_()
        self._action_history.zero_()
        self._observation.zero_()
        self._position_error_history.zero_()
        self._velocity_history.zero_()
        self._last_torque.zero_()
        self._phase.assign([self._initial_phase])
        return self.simulation.view(last_motor_torque=self.last_torque)
