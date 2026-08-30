"""Run a robot-selected ONNX gait policy through Newton/Kamino."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import torch
from newton.viewer import ViewerGL

from control.actuator import Actuator
from control.config import default_asset_root, discover_robot_profiles, get_robot_profile
from control.controller import Controller
from control.policy import OnnxPolicy, ZeroActionPolicy
from control.state import RobotState
from sim.simulation import KaminoSimulation

_VIEWER_RENDER_HZ = 30
_HEADING_GAIN = 1.0
_MAX_POLICY_YAW_RATE = 2.0


class KeyInput(Protocol):
    def is_key_down(self, key: str) -> bool: ...


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class HeadingHold:
    """Convert desired yaw rate into rate plus wrapped heading feedback."""

    def __init__(self, policy_dt: float) -> None:
        if not math.isfinite(policy_dt) or policy_dt <= 0.0:
            raise ValueError("policy_dt must be finite and positive")
        self.policy_dt = policy_dt
        self.target_yaw: float | None = None

    def reset(self, current_yaw: float) -> None:
        if not math.isfinite(current_yaw):
            raise ValueError("current_yaw must be finite")
        self.target_yaw = _wrap_angle(current_yaw)

    def command(
        self,
        values: tuple[float, float, float],
        *,
        current_yaw: float,
    ) -> tuple[float, float, float]:
        if self.target_yaw is None:
            self.reset(current_yaw)
        if not all(math.isfinite(value) for value in (*values, current_yaw)):
            raise ValueError("heading command values must be finite")
        vx, vy, desired_yaw_rate = values
        if vx == 0.0 and vy == 0.0 and desired_yaw_rate == 0.0:
            self.reset(current_yaw)
            return values
        heading_error = _wrap_angle(self.target_yaw - current_yaw)
        corrected_yaw_rate = max(
            -_MAX_POLICY_YAW_RATE,
            min(_MAX_POLICY_YAW_RATE, desired_yaw_rate + _HEADING_GAIN * heading_error),
        )
        self.target_yaw = _wrap_angle(self.target_yaw + desired_yaw_rate * self.policy_dt)
        return vx, vy, corrected_yaw_rate


@dataclass(frozen=True)
class PlayArgs:
    asset_root: Path = default_asset_root()
    policy: str | None = None
    onnx: Path | None = None
    device: str = "cuda:0"
    headless: bool = False
    steps: int = 0
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    robot: str | None = None
    zero_action: bool = False


def parse_args(argv: Sequence[str] | None = None) -> PlayArgs:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=default_asset_root(),
        help="asset tree containing robots, policies, actuator, and USD files",
    )
    parser.add_argument("--policy", help="dated policy set from policies/policies.json")
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument(
        "--robot",
        help="robot-policy pair to play; omit to use the default manifest",
    )
    parser.add_argument("--zero-action", action="store_true")
    parsed = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if parsed.headless and parsed.steps <= 0:
        parser.error("--steps must be positive in headless mode")
    if parsed.steps < 0:
        parser.error("--steps must be nonnegative")
    if not all(math.isfinite(value) for value in (parsed.vx, parsed.vy, parsed.yaw_rate)):
        parser.error("--vx, --vy, and --yaw-rate must be finite")
    return PlayArgs(
        asset_root=parsed.assets_root.expanduser().resolve(),
        policy=parsed.policy,
        onnx=parsed.onnx,
        device=parsed.device,
        headless=parsed.headless,
        steps=parsed.steps,
        vx=parsed.vx,
        vy=parsed.vy,
        yaw_rate=parsed.yaw_rate,
        robot=parsed.robot,
        zero_action=parsed.zero_action,
    )


def keyboard_command(viewer: KeyInput) -> tuple[float, float, float]:
    forward = 1.0 if viewer.is_key_down("i") else (-1.0 if viewer.is_key_down("k") else 0.0)
    lateral = 0.5 if viewer.is_key_down("j") else (-0.5 if viewer.is_key_down("l") else 0.0)
    yaw_rate = 1.0 if viewer.is_key_down("u") else (-1.0 if viewer.is_key_down("o") else 0.0)
    return forward, lateral, yaw_rate


def _validate_args(args: PlayArgs) -> None:
    if not isinstance(args, PlayArgs):
        raise TypeError("args must be a PlayArgs")
    if not isinstance(args.steps, int) or isinstance(args.steps, bool):
        raise TypeError("steps must be an integer")
    if args.headless and args.steps <= 0:
        raise ValueError("steps must be positive in headless mode")
    if args.steps < 0:
        raise ValueError("steps must be nonnegative")
    if not isinstance(args.asset_root, Path):
        raise TypeError("asset_root must be a pathlib.Path")
    if args.robot is not None and (not isinstance(args.robot, str) or not args.robot):
        raise ValueError("robot must be a non-empty string or None")
    if args.policy is not None and (not isinstance(args.policy, str) or not args.policy):
        raise ValueError("policy must be a non-empty string or None")
    if not all(math.isfinite(value) for value in (args.vx, args.vy, args.yaw_rate)):
        raise ValueError("vx, vy, and yaw_rate must be finite")


@dataclass
class _Metrics:
    initial_xy: torch.Tensor
    completed_steps: int = 0
    final_view: RobotState | None = None
    max_loop_error: torch.Tensor | None = None
    max_action: torch.Tensor | None = None
    elapsed_seconds: float = 0.0
    rendered_frames: int = 0
    _started_at: float | None = None

    def start(self) -> None:
        self._started_at = perf_counter()

    def finish(self) -> None:
        if self._started_at is None:
            raise RuntimeError("run metrics were not started")
        self.elapsed_seconds = max(0.0, perf_counter() - self._started_at)
        self._started_at = None

    def update(self, view: RobotState, action: torch.Tensor) -> None:
        self.completed_steps += 1
        self.final_view = view
        loop_error = view.loop_position_error.abs().amax()
        action_magnitude = action.abs().amax()
        self.max_loop_error = (
            loop_error if self.max_loop_error is None else torch.maximum(self.max_loop_error, loop_error)
        )
        self.max_action = (
            action_magnitude if self.max_action is None else torch.maximum(self.max_action, action_magnitude)
        )


def _command(values: tuple[float, float, float], device: torch.device) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.float32, device=device)


def _base_yaw(view: RobotState) -> float:
    x, y, z, w = (float(value.item()) for value in view.base_pose[0, 3:7])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _run_headless(
    controller: Controller,
    args: PlayArgs,
    metrics: _Metrics,
    heading: HeadingHold,
    initial_view: RobotState,
) -> None:
    desired = (args.vx, args.vy, args.yaw_rate)
    view = initial_view
    metrics.start()
    try:
        with torch.inference_mode():
            for _ in range(args.steps):
                values = heading.command(desired, current_yaw=_base_yaw(view))
                command = _command(values, controller.device)
                view = controller.step(command)
                metrics.update(view, controller.current_action)
    finally:
        metrics.finish()


def _run_viewer(
    controller: Controller,
    args: PlayArgs,
    viewer: ViewerGL,
    metrics: _Metrics,
    heading: HeadingHold,
    initial_view: RobotState,
) -> None:
    viewer.set_model(controller.simulation.model)
    fallback = (args.vx, args.vy, args.yaw_rate)
    view = initial_view
    reset_was_down = False
    policy_dt = controller.cfg.physics.policy_dt
    policy_hz = round(1.0 / policy_dt)
    render_phase = policy_hz - _VIEWER_RENDER_HZ
    metrics.start()
    try:
        with torch.inference_mode():
            while viewer.is_running() and (args.steps == 0 or metrics.completed_steps < args.steps):
                if not viewer.should_step():
                    continue
                reset_is_down = bool(viewer.is_key_down("p"))
                if reset_is_down and not reset_was_down:
                    view = controller.reset()
                    heading.reset(_base_yaw(view))
                reset_was_down = reset_is_down
                keys = keyboard_command(viewer)
                desired = keys if keys != (0.0, 0.0, 0.0) else fallback
                values = heading.command(desired, current_yaw=_base_yaw(view))
                view = controller.step(_command(values, controller.device))
                metrics.update(view, controller.current_action)
                render_phase += _VIEWER_RENDER_HZ
                if render_phase >= policy_hz:
                    viewer.begin_frame(metrics.completed_steps * policy_dt)
                    viewer.log_state(controller.simulation.current_state)
                    viewer.end_frame()
                    metrics.rendered_frames += 1
                    render_phase -= policy_hz
    finally:
        metrics.finish()


def _result(
    metrics: _Metrics,
    policy: OnnxPolicy | ZeroActionPolicy,
    actuator: Actuator,
    simulation: KaminoSimulation,
) -> dict[str, object]:
    if metrics.final_view is None:
        height = displacement = 0.0
    else:
        height = float(metrics.final_view.base_pose[0, 2].item())
        displacement = float(torch.linalg.vector_norm(metrics.final_view.base_pose[0, :2] - metrics.initial_xy).item())
    elapsed = metrics.elapsed_seconds
    policy_hz = metrics.completed_steps / elapsed if elapsed > 0.0 else 0.0
    result: dict[str, object] = {
        "completed_steps": metrics.completed_steps,
        "final_base_height": height,
        "final_xy_displacement": displacement,
        "max_abs_loop_position_error": 0.0 if metrics.max_loop_error is None else float(metrics.max_loop_error.item()),
        "max_abs_action": 0.0 if metrics.max_action is None else float(metrics.max_action.item()),
        "actuator_path": str(actuator.path),
        "actuator_sha256": actuator.sha256,
        "onnx_path": None if policy.path is None else str(policy.path),
        "onnx_sha256": policy.sha256,
        "elapsed_seconds": elapsed,
        "rendered_frames": metrics.rendered_frames,
        "render_fps": metrics.rendered_frames / elapsed if elapsed > 0.0 else 0.0,
        "policy_hz": policy_hz,
        "physics_fps": policy_hz * simulation.cfg.physics.policy_decimation,
        "real_time_factor": policy_hz * simulation.cfg.physics.policy_dt,
        "cuda_graph_enabled": simulation.cuda_graph_enabled,
        "robot": simulation.robot_name,
        "controller": policy.mode,
    }
    non_finite = [name for name, value in result.items() if isinstance(value, float) and not math.isfinite(value)]
    if non_finite:
        raise FloatingPointError(f"non-finite run outputs: {', '.join(non_finite)}")
    return result


def play(args: PlayArgs) -> dict[str, object]:
    _validate_args(args)
    profiles, default_robot = discover_robot_profiles(args.asset_root, policy=args.policy)
    profile = get_robot_profile(
        default_robot if args.robot is None else args.robot,
        profiles,
        asset_root=args.asset_root,
    )
    device = torch.device(args.device)
    policy_path = profile.policy.path if args.onnx is None else args.onnx
    policy = (
        ZeroActionPolicy(device, profile.observation.input_width, profile.action_dim)
        if args.zero_action
        else OnnxPolicy(policy_path, profile)
    )
    actuator = Actuator(profile, device)
    simulation: KaminoSimulation | None = None
    viewer: ViewerGL | None = None
    try:
        simulation = KaminoSimulation(
            profile,
            1,
            args.device,
            load_visual_shapes=not args.headless,
            use_cuda_graph=True,
        )
        if simulation.device.type == "cuda" and not simulation.cuda_graph_enabled:
            raise RuntimeError("CUDA graph capture is required for Kamino control")
        controller = Controller(simulation, profile, policy, actuator)
        initial = controller.reset()
        heading = HeadingHold(profile.config.physics.policy_dt)
        heading.reset(_base_yaw(initial))
        metrics = _Metrics(initial_xy=initial.base_pose[0, :2].clone())
        if args.headless:
            _run_headless(controller, args, metrics, heading, initial)
        else:
            viewer = ViewerGL(vsync=False)
            _run_viewer(controller, args, viewer, metrics, heading, initial)
        return _result(metrics, policy, actuator, simulation)
    finally:
        if viewer is not None:
            viewer.close()


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(play(parse_args(argv)), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
