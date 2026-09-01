"""Device-resident public-API Kamino simulation for supported robots."""

from __future__ import annotations

from dataclasses import dataclass

import newton
import torch
import warp as wp

from control.config import RobotProfile, SolverConfig
from control.state import RobotState
from sim.model import build_batched_model


@dataclass(frozen=True)
class _StateTensors:
    body_pose: torch.Tensor
    body_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor


def _quaternion_rotate_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by XYZW quaternions without leaving their Torch device."""
    twice_cross = 2.0 * torch.linalg.cross(quaternion[..., :3], vector, dim=-1)
    return vector + quaternion[..., 3:4] * twice_cross + torch.linalg.cross(quaternion[..., :3], twice_cross, dim=-1)


def _apply_solver_config(target: newton.solvers.SolverKamino.Config, source: SolverConfig) -> None:
    """Apply one manifest-declared DVI configuration."""
    target.sparse_jacobian = source.sparse_jacobian
    target.sparse_dynamics = source.sparse_dynamics
    target.integrator = source.integrator
    target.constraints.alpha = source.constraint_alpha
    target.constraints.beta = source.constraint_beta
    target.constraints.gamma = source.constraint_gamma
    target.dynamics.linear_solver_type = source.linear_solver
    target.dynamics.linear_solver_kwargs = {"maxiter": source.linear_solver_max_iterations}
    target.dynamics.preconditioning = source.preconditioning
    target.dvi.bilateral_solver_type = source.bilateral_solver
    target.dvi.bilateral_solver_kwargs = {"parallel_factorization": source.parallel_factorization}
    target.dvi.tolerance = source.tolerance
    target.dvi.regularization = source.regularization
    target.dvi.max_alternating_iterations = source.alternating_iterations
    target.dvi.inequality_sweeps_per_iteration = source.inequality_sweeps
    target.dvi.bilateral_solve_interval = source.bilateral_solve_interval
    target.dvi.warmstart_mode = source.warmstart_mode
    target.dvi.contact_warmstart_method = source.contact_warmstart_method
    target.materials.friction_mix_mode = source.friction_mix_mode
    target.materials.restitution_mix_mode = source.restitution_mix_mode


class KaminoSimulation:
    """Batched robot stepping and closed-chain reset through Newton's public API."""

    def __init__(
        self,
        profile: RobotProfile,
        num_envs: int,
        device: str | torch.device,
        *,
        load_visual_shapes: bool = True,
        use_cuda_graph: bool = True,
    ) -> None:
        if not isinstance(profile, RobotProfile):
            raise TypeError("profile must be a RobotProfile")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive; got {num_envs}")

        self.profile = profile
        self.cfg = profile.config
        self.robot_name = profile.name
        self.robot_label = profile.label
        self.num_envs = num_envs
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.model, self.layout = build_batched_model(
            profile,
            num_envs,
            str(self.device),
            load_visual_shapes=load_visual_shapes,
        )

        solver_cfg = newton.solvers.SolverKamino.Config.from_model(self.model, dynamics_solver="dvi")
        solver_cfg.use_fk_solver = True
        solver_cfg.use_collision_detector = False
        _apply_solver_config(solver_cfg, self.cfg.solver)
        self.solver = newton.solvers.SolverKamino(self.model, config=solver_cfg)

        self.state_in, self.state_out = self.model.state(), self.model.state()
        self._state_0, self._state_1 = self.state_in, self.state_out
        self._states = (self._state_0, self._state_1)
        self._active_state_index = 0
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        # Wrap each public Warp array exactly once. State gathers below remain on device.
        self._state_in_tensors = self._wrap_state(self.state_in)
        self._state_out_tensors = self._wrap_state(self.state_out)
        self.control_joint_force = wp.to_torch(self.control.joint_f)
        self._joint_parent = wp.to_torch(self.model.joint_parent).long()
        self._joint_child = wp.to_torch(self.model.joint_child).long()
        self._joint_parent_frame = wp.to_torch(self.model.joint_X_p)
        self._joint_child_frame = wp.to_torch(self.model.joint_X_c)
        self._shape_body = wp.to_torch(self.model.shape_body).long()
        self._shape_type = wp.to_torch(self.model.shape_type).long()
        self._shape_scale = wp.to_torch(self.model.shape_scale)
        self._use_cuda_graph = bool(use_cuda_graph and self.device.type == "cuda")
        self._cuda_graphs: tuple[wp.Graph, wp.Graph] | None = None
        loop_parents = self._joint_parent[self.layout.loop_joint_indices]
        loop_children = self._joint_child[self.layout.loop_joint_indices]
        if (loop_parents < 0).any() or (loop_children < 0).any():
            raise ValueError(f"{self.robot_label} loop-closing joints must connect two bodies")

        self.last_motor_torque = torch.zeros(num_envs, self.cfg.action_dim, device=self.device)
        self._home_motor_q = torch.tensor(self.cfg.actuator.home_position, dtype=torch.float32, device=self.device)
        self._actuator_q = torch.empty(num_envs, self.cfg.action_dim, device=self.device)
        self._actuator_u = torch.zeros_like(self._actuator_q)
        self._base_q = torch.zeros(num_envs, 7, device=self.device)
        self._base_q[:, 6] = 1.0
        self._base_u = torch.zeros(num_envs, 6, device=self.device)
        self._world_mask = torch.zeros(num_envs + 1, dtype=torch.bool, device=self.device)
        self._success_0 = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._success_1 = torch.zeros_like(self._success_0)

        # Pinned Newton expects flat actuator arrays, while Torch retains batched buffers.
        self._actuator_q_wp = wp.from_torch(self._actuator_q.reshape(-1), dtype=wp.float32)
        self._actuator_u_wp = wp.from_torch(self._actuator_u.reshape(-1), dtype=wp.float32)
        self._base_q_wp = wp.from_torch(self._base_q, dtype=wp.transformf)
        self._base_u_wp = wp.from_torch(self._base_u, dtype=wp.spatial_vectorf)
        self._world_mask_wp = wp.from_torch(self._world_mask, dtype=wp.bool)
        self._success_0_wp = wp.from_torch(self._success_0, dtype=wp.bool)
        self._success_1_wp = wp.from_torch(self._success_1, dtype=wp.bool)

        foot_radii = self._resolve_foot_radii()

        self.nominal_base_pose = torch.zeros(num_envs, 7, device=self.device)
        self.nominal_base_pose[:, 6] = 1.0
        self._actuator_q.copy_(self._home_motor_q)
        provisional = self._nominal_reset_config()
        self._reset_both(provisional, torch.ones(num_envs, dtype=torch.bool, device=self.device))
        if not (self._success_0 & self._success_1).all():
            raise RuntimeError(f"Kamino failed to derive the nominal {self.robot_label} forward-kinematics pose")

        foot_heights = self.view().foot_pose[:, :, 2]
        offsets = foot_radii - foot_heights
        if (offsets.amax(dim=1) - offsets.amin(dim=1)).max() > 1.0e-4:
            raise ValueError(f"nominal {self.robot_label} foot contacts are not coplanar")
        self.nominal_base_pose[:, 2] = profile.reset_base_height
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=self.device))

    @staticmethod
    def _wrap_state(state: newton.State) -> _StateTensors:
        return _StateTensors(
            body_pose=wp.to_torch(state.body_q),
            body_velocity=wp.to_torch(state.body_qd),
            joint_position=wp.to_torch(state.joint_q),
            joint_velocity=wp.to_torch(state.joint_qd),
        )

    @property
    def current_state(self) -> newton.State:
        """Return the state that will be consumed by the next physics step."""
        return self.state_in

    @property
    def cuda_graph_enabled(self) -> bool:
        """Report whether fixed-buffer simulation steps use captured CUDA graphs."""
        return self._cuda_graphs is not None

    def _current_tensors(self) -> _StateTensors:
        if self.state_in is self._state_0:
            return self._state_in_tensors
        if self.state_in is self._state_1:
            return self._state_out_tensors
        raise RuntimeError("invalid Kamino ping-pong state")

    def _simulate_once(self, state_in: newton.State, state_out: newton.State) -> None:
        """Record or execute one fixed-buffer collision and DVI step."""
        state_in.clear_forces()
        self.collision_pipeline.collide(state_in, self.contacts)
        self.solver.step(state_in, state_out, self.control, self.contacts, self.cfg.physics.dt)
        self.solver.update_contacts(self.contacts, state_out)

    def simulate_once(self) -> None:
        """Advance one physics step for a surrounding device-resident controller."""
        self._simulate_once(self.state_in, self.state_out)
        self.state_in, self.state_out = self.state_out, self.state_in
        self._active_state_index = 1 - self._active_state_index

    def _capture_cuda_graphs(self, reset_cfg: newton.solvers.SolverKamino.ResetConfig) -> None:
        """Capture one public-API Warp graph for each state-buffer direction."""
        state_0, state_1 = self._states
        all_worlds = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Initialize every lazy DVI/collision allocation before graph recording.
        self._simulate_once(state_0, state_1)
        self._reset_both(reset_cfg, all_worlds)
        with wp.ScopedCapture() as capture_0:
            self._simulate_once(state_0, state_1)
        with wp.ScopedCapture() as capture_1:
            self._simulate_once(state_1, state_0)
        self._cuda_graphs = (capture_0.graph, capture_1.graph)

        self._reset_both(reset_cfg, all_worlds)
        self.control_joint_force.zero_()
        self._active_state_index = 0
        self.state_in, self.state_out = state_0, state_1

    def _resolve_foot_radii(self) -> torch.Tensor:
        shape_body = self._shape_body.cpu().tolist()
        shape_type = self._shape_type.cpu().tolist()
        radii = torch.empty(self.num_envs, len(self.profile.model.foot_bodies), device=self.device)
        for world, foot_bodies in enumerate(self.layout.foot_body_indices.cpu().tolist()):
            for foot, body in enumerate(foot_bodies):
                matches = [
                    index
                    for index, owner in enumerate(shape_body)
                    if owner == body and shape_type[index] == int(newton.GeoType.SPHERE)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"{self.robot_label} foot body {body} in world {world} has {len(matches)} spherical shapes; "
                        "expected one"
                    )
                shape = matches[0]
                radii[world, foot] = self._shape_scale[shape, 0]
        return radii

    def _validate_mask(self, mask: torch.Tensor) -> None:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor")
        if mask.shape != (self.num_envs,) or mask.dtype != torch.bool or mask.device != self.device:
            raise ValueError(f"mask must be a boolean tensor with shape ({self.num_envs},) on {self.device}")

    def _set_world_mask(self, mask: torch.Tensor) -> None:
        self._world_mask[: self.num_envs].copy_(mask)
        self._world_mask[self.num_envs] = False

    def _nominal_reset_config(self) -> newton.solvers.SolverKamino.ResetConfig:
        return newton.solvers.SolverKamino.ResetConfig(
            body_poses=newton.solvers.SolverKamino.ResetConfig.FromActuatorQ(self._actuator_q_wp),
            body_velocities=newton.solvers.SolverKamino.ResetConfig.FromActuatorU(self._actuator_u_wp),
            base_pose=newton.solvers.SolverKamino.ResetConfig.FromBaseQ(self._base_q_wp),
            base_velocity=newton.solvers.SolverKamino.ResetConfig.FromBaseU(self._base_u_wp),
        )

    def _reset_both(self, reset_cfg: newton.solvers.SolverKamino.ResetConfig, mask: torch.Tensor) -> None:
        self._set_world_mask(mask)
        self._success_0.zero_()
        self._success_1.zero_()
        self.solver.reset(
            self.state_in,
            world_mask=self._world_mask_wp,
            config=reset_cfg,
            success_mask=self._success_0_wp,
        )
        self.solver.reset(
            self.state_out,
            world_mask=self._world_mask_wp,
            config=reset_cfg,
            success_mask=self._success_1_wp,
        )

    def _loop_position_error(self, body_pose: torch.Tensor) -> torch.Tensor:
        joints = self.layout.loop_joint_indices
        parents = self._joint_parent[joints]
        children = self._joint_child[joints]
        parent_pose = body_pose[parents]
        child_pose = body_pose[children]
        parent_frame = self._joint_parent_frame[joints]
        child_frame = self._joint_child_frame[joints]
        parent_anchor = parent_pose[..., :3] + _quaternion_rotate_xyzw(parent_pose[..., 3:], parent_frame[..., :3])
        child_anchor = child_pose[..., :3] + _quaternion_rotate_xyzw(child_pose[..., 3:], child_frame[..., :3])
        return torch.linalg.vector_norm(parent_anchor - child_anchor, dim=-1)

    def view(self, *, last_motor_torque: torch.Tensor | None = None) -> RobotState:
        """Gather a device-resident view of the current state."""
        tensors = self._current_tensors()
        return RobotState(
            base_pose=tensors.body_pose[self.layout.base_body_indices],
            base_velocity=tensors.body_velocity[self.layout.base_body_indices],
            motor_position=tensors.joint_position[self.layout.motor_coord_indices],
            motor_velocity=tensors.joint_velocity[self.layout.motor_dof_indices],
            foot_pose=tensors.body_pose[self.layout.foot_body_indices],
            foot_velocity=tensors.body_velocity[self.layout.foot_body_indices],
            loop_position_error=self._loop_position_error(tensors.body_pose),
            applied_motor_torque=self.last_motor_torque if last_motor_torque is None else last_motor_torque,
        )

    def reset(self, mask: torch.Tensor) -> torch.Tensor:
        """Reset selected worlds to the exact grounded home pose, retrying failed FK once."""
        self._validate_mask(mask)
        self._actuator_q.copy_(self._home_motor_q)
        self._actuator_u.zero_()
        self._base_q.copy_(self.nominal_base_pose)
        self._base_u.zero_()
        reset_cfg = self._nominal_reset_config()

        self._reset_both(reset_cfg, mask)
        initial_0 = self._success_0.clone()
        initial_1 = self._success_1.clone()
        retry = mask & ~(initial_0 & initial_1)
        self._reset_both(reset_cfg, retry)
        retry_success = self._success_0 & self._success_1
        success = (mask & initial_0 & initial_1) | (retry & retry_success)
        failed = retry & ~retry_success
        if failed.any():
            ids = failed.nonzero(as_tuple=False).flatten()
            raise RuntimeError(
                f"Kamino {self.robot_label} reset failed twice; "
                f"world_ids={ids.tolist()}, motor_positions={self._actuator_q[ids].tolist()}, "
                f"base_inputs={self._base_q[ids].tolist()}"
            )

        self.control_joint_force.zero_()
        self.last_motor_torque[success] = 0.0
        if self._use_cuda_graph and self._cuda_graphs is None:
            self._capture_cuda_graphs(reset_cfg)
        elif bool(mask.all()):
            self._active_state_index = 0
            self.state_in, self.state_out = self._states
        return success

    def step_motor_torques(self, torque: torch.Tensor, *, validate_finite: bool = True) -> RobotState:
        """Clamp and route motor efforts, then advance one physics step."""
        if not isinstance(torque, torch.Tensor):
            raise TypeError("torque must be a torch.Tensor")
        expected = (self.num_envs, self.cfg.action_dim)
        if torque.shape != expected or torque.device != self.device:
            raise ValueError(f"torque must have shape {expected} on {self.device}")
        if validate_finite and not torch.isfinite(torque).all():
            raise FloatingPointError("torque must contain only finite values")

        self.last_motor_torque.copy_(torque).clamp_(
            -self.cfg.actuator.effort_limit,
            self.cfg.actuator.effort_limit,
        )
        self.control_joint_force.zero_()
        self.control_joint_force.index_copy_(
            0,
            self.layout.motor_dof_indices.reshape(-1),
            self.last_motor_torque.reshape(-1),
        )

        if self._cuda_graphs is None:
            self._simulate_once(self.state_in, self.state_out)
            self.state_in, self.state_out = self.state_out, self.state_in
            self._active_state_index = 1 - self._active_state_index
        else:
            wp.capture_launch(self._cuda_graphs[self._active_state_index])
            self._active_state_index = 1 - self._active_state_index
            self.state_in = self._states[self._active_state_index]
            self.state_out = self._states[1 - self._active_state_index]
        return self.view()
