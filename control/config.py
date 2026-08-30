"""Load robot assets and dated policy contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PhysicsConfig:
    dt: float
    policy_decimation: int
    gravity: float

    @property
    def policy_dt(self) -> float:
        return self.dt * self.policy_decimation


@dataclass(frozen=True)
class SolverConfig:
    sparse_jacobian: bool
    sparse_dynamics: bool
    integrator: str
    constraint_alpha: float
    constraint_beta: float
    constraint_gamma: float
    linear_solver: str
    linear_solver_max_iterations: int
    preconditioning: bool
    bilateral_solver: str
    parallel_factorization: bool
    tolerance: float
    regularization: float
    alternating_iterations: int
    inequality_sweeps: int
    bilateral_solve_interval: int
    warmstart_mode: str
    contact_warmstart_method: str
    friction_mix_mode: str
    restitution_mix_mode: str


@dataclass(frozen=True)
class ActuatorConfig:
    home_position: tuple[float, ...]
    action_scale: float
    effort_limit: float
    velocity_limit: float
    saturation_effort: float
    simulation_velocity_limit: float
    armature: float
    passive_damping: float


@dataclass(frozen=True)
class ControlConfig:
    physics: PhysicsConfig
    solver: SolverConfig
    actuator: ActuatorConfig
    action_dim: int
    action_clip: float


@dataclass(frozen=True)
class ModelConfig:
    body_count: int
    joint_count: int
    articulation_count: int
    base_body: str
    motor_joints: tuple[str, ...]
    foot_bodies: tuple[str, ...]
    loop_joints: tuple[str, ...]
    dynamic_friction: float
    ground_friction: float
    restitution: float
    enable_self_collisions: bool


@dataclass(frozen=True)
class ClockConfig:
    frequency_hz: float
    phase_offsets: tuple[float, ...]
    command_deadband: tuple[float, float, float]
    standstill_values: tuple[float, ...]
    advance_before_observation: bool


@dataclass(frozen=True)
class ObservationFieldConfig:
    name: str
    width: int
    scale: float


@dataclass(frozen=True)
class ObservationConfig:
    history_length: int
    history_order: str
    policy_motor_order: tuple[str, ...]
    gravity_direction: tuple[float, float, float]
    clock: ClockConfig
    fields: tuple[ObservationFieldConfig, ...]

    @property
    def frame_width(self) -> int:
        return sum(field.width for field in self.fields)

    @property
    def input_width(self) -> int:
        return self.history_length * self.frame_width

    @property
    def action_history_length(self) -> int:
        return sum(field.name in {"current_action", "previous_action"} for field in self.fields)


@dataclass(frozen=True)
class PolicyTensorConfig:
    name: str
    dtype: str
    batch: int | str
    width: int

    @property
    def shape(self) -> tuple[int | str, int]:
        return self.batch, self.width


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    path: Path
    sha256: str
    input: PolicyTensorConfig
    output: PolicyTensorConfig


@dataclass(frozen=True)
class RobotProfile:
    name: str
    label: str
    usd_path: Path
    usd_sha256: str
    policy: PolicyConfig
    actuator_path: Path
    actuator_sha256: str
    reset_base_height: float
    model: ModelConfig
    observation: ObservationConfig
    config: ControlConfig

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    @property
    def actuator(self) -> ActuatorConfig:
        return self.config.actuator


def default_asset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "assets"


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid configuration {path}: root must be an object")
    return value


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _strings(value: Any, field: str) -> tuple[str, ...]:
    items = _array(value, field)
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{field} must contain non-empty strings")
    return tuple(items)


def _numbers(value: Any, width: int, field: str) -> tuple[float, ...]:
    items = _array(value, field)
    if len(items) != width or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in items):
        raise ValueError(f"{field} must contain {width} numbers")
    return tuple(float(item) for item in items)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _asset_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{field} escapes {root}")
    return path


def _digest(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _policy_tensor(value: Any, field: str) -> PolicyTensorConfig:
    data = _object(value, field)
    name = data.get("name")
    dtype = data.get("dtype")
    batch = data.get("batch")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{field}.name must be a non-empty string")
    if dtype != "tensor(float)":
        raise ValueError(f"{field}.dtype must be 'tensor(float)'")
    if isinstance(batch, bool) or not ((isinstance(batch, int) and batch > 0) or (isinstance(batch, str) and batch)):
        raise ValueError(f"{field}.batch must be a positive integer or symbolic name")
    return PolicyTensorConfig(
        name=name,
        dtype=dtype,
        batch=batch,
        width=_positive_int(data.get("width"), f"{field}.width"),
    )


def _observation(value: Any, *, action_dim: int, model: ModelConfig, field: str) -> ObservationConfig:
    data = _object(value, field)
    history = _object(data.get("history"), f"{field}.history")
    history_length = _positive_int(history.get("length"), f"{field}.history.length")
    history_order = history.get("order")
    if history_order != "oldest_to_newest":
        raise ValueError(f"{field}.history.order must be 'oldest_to_newest'")

    motor_order = _strings(data.get("motor_order"), f"{field}.motor_order")
    if len(motor_order) != action_dim or set(motor_order) != set(model.motor_joints):
        raise ValueError(f"{field}.motor_order must contain every motor joint exactly once")
    gravity = _numbers(data.get("gravity_direction"), 3, f"{field}.gravity_direction")

    clock_data = _object(data.get("clock"), f"{field}.clock")
    clock = ClockConfig(
        frequency_hz=_number(clock_data.get("frequency_hz"), f"{field}.clock.frequency_hz", positive=True),
        phase_offsets=_numbers(clock_data.get("phase_offsets"), len(model.foot_bodies), f"{field}.clock.phase_offsets"),
        command_deadband=_numbers(clock_data.get("command_deadband"), 3, f"{field}.clock.command_deadband"),
        standstill_values=_numbers(
            clock_data.get("standstill_values"), len(model.foot_bodies), f"{field}.clock.standstill_values"
        ),
        advance_before_observation=clock_data.get("advance_before_observation"),
    )
    if not isinstance(clock.advance_before_observation, bool):
        raise ValueError(f"{field}.clock.advance_before_observation must be boolean")
    if any(value < 0.0 for value in clock.command_deadband):
        raise ValueError(f"{field}.clock.command_deadband values must be nonnegative")

    expected_widths = {
        "gait_clock": len(model.foot_bodies),
        "projected_gravity": 3,
        "base_angular_velocity": 3,
        "velocity_command": 3,
        "motor_position_error": action_dim,
        "motor_velocity": action_dim,
        "current_action": action_dim,
        "previous_action": action_dim,
    }
    fields: list[ObservationFieldConfig] = []
    for index, item in enumerate(_array(data.get("fields"), f"{field}.fields")):
        item_data = _object(item, f"{field}.fields[{index}]")
        name = item_data.get("name")
        if name not in expected_widths:
            raise ValueError(f"{field}.fields[{index}].name is unsupported: {name!r}")
        width = _positive_int(item_data.get("width"), f"{field}.fields[{index}].width")
        if width != expected_widths[name]:
            raise ValueError(f"{field}.{name} width must be {expected_widths[name]}")
        fields.append(
            ObservationFieldConfig(
                name=name,
                width=width,
                scale=_number(item_data.get("scale"), f"{field}.fields[{index}].scale"),
            )
        )
    names = tuple(item.name for item in fields)
    if len(names) != len(expected_widths) or set(names) != set(expected_widths):
        raise ValueError(f"{field}.fields must contain every supported observation field exactly once")
    return ObservationConfig(
        history_length=history_length,
        history_order=history_order,
        policy_motor_order=motor_order,
        gravity_direction=gravity,  # type: ignore[arg-type]
        clock=clock,
        fields=tuple(fields),
    )


def discover_robot_profiles(
    asset_root: str | Path,
    *,
    policy: str | None = None,
) -> tuple[dict[str, RobotProfile], str]:
    """Build runnable robot profiles for one named policy set."""

    root = Path(asset_root).resolve()
    robot_path = root / "robots" / "robots.json"
    policy_root = root / "policies"
    policy_path = policy_root / "policies.json"
    robots_data = _read_object(robot_path)
    policies_data = _read_object(policy_path)

    if robots_data.get("schema_version") != 1 or policies_data.get("schema_version") != 1:
        raise ValueError("robots.json and policies.json must use schema_version 1")
    policy_sets = _object(policies_data.get("sets"), "sets")
    selected = policies_data.get("default") if policy is None else policy
    if not isinstance(selected, str) or selected not in policy_sets:
        choices = ", ".join(policy_sets)
        raise ValueError(f"unknown policy {selected!r}; expected one of: {choices}")
    selected_data = _object(policy_sets[selected], f"sets.{selected}")

    model_data = _object(robots_data.get("model"), "model")
    model_common = dict(model_data)
    for key in ("motor_joints", "foot_bodies", "loop_joints"):
        model_common[key] = _strings(model_data.get(key), f"model.{key}")
    action_dim = len(model_common["motor_joints"])
    if action_dim == 0:
        raise ValueError("model.motor_joints must not be empty")

    action_data = _object(selected_data.get("action"), f"sets.{selected}.action")
    if _positive_int(action_data.get("width"), f"sets.{selected}.action.width") != action_dim:
        raise ValueError(f"sets.{selected}.action.width must match the robot motor count")
    action_scale = _number(action_data.get("scale"), f"sets.{selected}.action.scale", positive=True)
    action_clip = _number(action_data.get("clip"), f"sets.{selected}.action.clip", positive=True)

    control_data = _object(selected_data.get("control"), f"sets.{selected}.control")
    physics_data = dict(_object(robots_data.get("physics"), "physics"))
    physics_data["policy_decimation"] = _positive_int(
        control_data.get("decimation"), f"sets.{selected}.control.decimation"
    )
    physics = PhysicsConfig(**physics_data)
    solver = SolverConfig(**_object(robots_data.get("solver"), "solver"))
    reset_base_height = _number(robots_data.get("reset_base_height"), "reset_base_height", positive=True)
    files = _object(robots_data.get("files"), "files")
    actuator_path = _asset_path(root, files.get("actuator"), "files.actuator")
    actuator_sha256 = _digest(files.get("actuator_sha256"), "files.actuator_sha256")
    actuator_common = _object(robots_data.get("actuator"), "actuator")
    robot_entries = _object(robots_data.get("robots"), "robots")
    default_robot_manifest = robots_data.get("default")
    if not isinstance(default_robot_manifest, str) or default_robot_manifest not in robot_entries:
        raise ValueError(f"default robot {default_robot_manifest!r} is not configured")

    policy_robots = _object(selected_data.get("robots"), f"sets.{selected}.robots")
    default_robot = selected_data.get("default_robot")
    if not isinstance(default_robot, str) or default_robot not in policy_robots:
        raise ValueError(f"sets.{selected}.default_robot must name a registered robot")

    profiles: dict[str, RobotProfile] = {}
    for name, policy_value in policy_robots.items():
        if name not in robot_entries:
            raise ValueError(f"policy {selected!r} references unknown robot {name!r}")
        robot = _object(robot_entries[name], f"robots.{name}")
        model = ModelConfig(enable_self_collisions=robot.get("enable_self_collisions"), **model_common)
        observation = _observation(
            selected_data.get("observation"),
            action_dim=action_dim,
            model=model,
            field=f"sets.{selected}.observation",
        )
        policy_data = _object(policy_value, f"sets.{selected}.robots.{name}")
        policy_input = _policy_tensor(policy_data.get("input"), f"sets.{selected}.robots.{name}.input")
        policy_output = _policy_tensor(policy_data.get("output"), f"sets.{selected}.robots.{name}.output")
        if policy_input.width != observation.input_width:
            raise ValueError(f"{selected}/{name} input width must equal the configured observation width")
        if policy_output.width != action_dim:
            raise ValueError(f"{selected}/{name} output width must equal the robot motor count")
        home_position = _numbers(robot.get("home_position"), action_dim, f"robots.{name}.home_position")
        actuator = ActuatorConfig(home_position=home_position, action_scale=action_scale, **actuator_common)
        profiles[name] = RobotProfile(
            name=name,
            label=robot.get("label"),
            usd_path=_asset_path(root, robot.get("usd"), f"robots.{name}.usd"),
            usd_sha256=_digest(robot.get("usd_sha256"), f"robots.{name}.usd_sha256"),
            policy=PolicyConfig(
                name=selected,
                path=_asset_path(policy_root, policy_data.get("model"), f"sets.{selected}.robots.{name}.model"),
                sha256=_digest(policy_data.get("sha256"), f"sets.{selected}.robots.{name}.sha256"),
                input=policy_input,
                output=policy_output,
            ),
            actuator_path=actuator_path,
            actuator_sha256=actuator_sha256,
            reset_base_height=reset_base_height,
            model=model,
            observation=observation,
            config=ControlConfig(
                physics=physics,
                solver=solver,
                actuator=actuator,
                action_dim=action_dim,
                action_clip=action_clip,
            ),
        )
    return profiles, default_robot


def get_robot_profile(
    name: str,
    profiles: Mapping[str, RobotProfile],
    *,
    asset_root: str | Path | None = None,
) -> RobotProfile:
    try:
        return profiles[name]
    except KeyError as error:
        choices = ", ".join(profiles)
        root_context = "" if asset_root is None else f"; asset root: {Path(asset_root).resolve()}"
        raise ValueError(f"unknown robot {name!r}; expected one of: {choices}{root_context}") from error


def _verify(kind: str, profile: RobotProfile, path: Path, expected: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{profile.name} {kind} does not exist: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{profile.name} {kind} SHA-256 mismatch: expected {expected}, got {digest}")
    return path


def verify_robot_asset(profile: RobotProfile) -> Path:
    return _verify("robot asset", profile, profile.usd_path, profile.usd_sha256)


def verify_policy_asset(profile: RobotProfile) -> Path:
    return _verify("policy", profile, profile.policy.path, profile.policy.sha256)


def verify_actuator_asset(profile: RobotProfile) -> Path:
    return _verify("actuator network", profile, profile.actuator_path, profile.actuator_sha256)
