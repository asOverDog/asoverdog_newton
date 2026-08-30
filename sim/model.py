"""Public-API Newton loading and semantic indexing for supported robots."""

from __future__ import annotations

from dataclasses import dataclass

import newton
import numpy as np
import torch

from control.config import RobotProfile, verify_robot_asset


@dataclass(frozen=True)
class RobotLayout:
    """Semantic indices into one imported robot builder."""

    motor_joint_names: tuple[str, ...]
    base_body_index: int
    motor_joint_indices: tuple[int, ...]
    motor_coord_indices: tuple[int, ...]
    motor_dof_indices: tuple[int, ...]
    foot_body_indices: tuple[int, ...]
    loop_joint_indices: tuple[int, ...]


@dataclass(frozen=True)
class BatchedLayout:
    """Device-resident semantic indices into a replicated Newton model."""

    num_worlds: int
    motor_joint_names: tuple[str, ...]
    base_body_indices: torch.Tensor
    motor_joint_indices: torch.Tensor
    motor_coord_indices: torch.Tensor
    motor_dof_indices: torch.Tensor
    foot_body_indices: torch.Tensor
    loop_joint_indices: torch.Tensor


def _short_label(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _unique_short_label(labels: list[str], name: str, entity: str, robot_label: str) -> int:
    """Return the unique index whose final path component equals ``name``."""
    matches = [index for index, label in enumerate(labels) if _short_label(label) == name]
    if len(matches) != 1:
        raise ValueError(f"{robot_label} {entity} label {name!r} matched {len(matches)} entries; expected exactly one")
    return matches[0]


def _joint_width(starts: list[int], index: int, total: int) -> int:
    end = starts[index + 1] if index + 1 < len(starts) else total
    return end - starts[index]


def _named_indices(labels: list[str], names: tuple[str, ...], entity: str, robot_label: str) -> tuple[int, ...]:
    return tuple(_unique_short_label(labels, name, entity, robot_label) for name in names)


def _quaternion_matrix_xyzw(quaternion: object) -> np.ndarray:
    """Return a body-local rotation matrix for one finite XYZW quaternion."""
    value = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(value)
    if value.shape != (4,) or not np.all(np.isfinite(value)) or norm <= 1.0e-12:
        raise ValueError("visual shape quaternion must contain four finite values with nonzero norm")
    x, y, z, w = value / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _add_merged_visual_shapes(builder: newton.ModelBuilder, visual_builder: newton.ModelBuilder) -> None:
    """Merge visual-only CAD meshes by body/material before finalization."""
    visible = int(newton.ShapeFlags.VISIBLE)
    colliding = int(newton.ShapeFlags.COLLIDE_SHAPES)
    groups: dict[tuple[object, ...], list[int]] = {}
    for shape_index in range(visual_builder.shape_count):
        flags = int(visual_builder.shape_flags[shape_index])
        if not flags & visible or flags & colliding or visual_builder.shape_type[shape_index] != newton.GeoType.MESH:
            continue
        mesh = visual_builder.shape_source[shape_index]
        key = (
            visual_builder.shape_body[shape_index],
            tuple(float(value) for value in visual_builder.shape_color[shape_index]),
            mesh.roughness,
            mesh.metallic,
        )
        groups.setdefault(key, []).append(shape_index)

    visual_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
        is_visible=True,
    )
    target_bodies = {label: index for index, label in enumerate(builder.body_label)}
    for group_index, ((source_body, color, roughness, metallic), shape_indices) in enumerate(groups.items()):
        body = target_bodies[visual_builder.body_label[source_body]]
        vertices_parts: list[np.ndarray] = []
        indices_parts: list[np.ndarray] = []
        normals_parts: list[np.ndarray] = []
        vertex_offset = 0
        all_have_normals = True
        for shape_index in shape_indices:
            source = visual_builder.shape_source[shape_index]
            scale = np.asarray(visual_builder.shape_scale[shape_index], dtype=np.float64)
            transform = visual_builder.shape_transform[shape_index]
            position = np.asarray(transform.p, dtype=np.float64)
            rotation = _quaternion_matrix_xyzw(transform.q)
            vertices = np.asarray(source.vertices, dtype=np.float64) * scale
            vertices_parts.append((vertices @ rotation.T + position).astype(np.float32))

            indices = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3).copy()
            if np.prod(scale) < 0.0:
                indices[:, [1, 2]] = indices[:, [2, 1]]
            indices_parts.append(indices.reshape(-1) + vertex_offset)
            vertex_offset += len(vertices)

            if source.normals is None:
                all_have_normals = False
            elif all_have_normals:
                normals = np.asarray(source.normals, dtype=np.float64) / scale
                normals = normals @ rotation.T
                lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                normals_parts.append((normals / np.maximum(lengths, 1.0e-20)).astype(np.float32))

        merged_mesh = newton.Mesh(
            np.concatenate(vertices_parts),
            np.concatenate(indices_parts),
            normals=np.concatenate(normals_parts) if all_have_normals else None,
            compute_inertia=False,
            color=color,
            roughness=roughness,
            metallic=metallic,
        )
        builder.add_shape_mesh(
            body=body,
            mesh=merged_mesh,
            cfg=visual_cfg,
            color=color,
            label=f"visual_batch_{group_index}",
        )


def _classify_layout(builder: newton.ModelBuilder, profile: RobotProfile) -> RobotLayout:
    """Validate one robot builder and derive all semantic indices."""
    robot_label = profile.label
    model = profile.model
    counts = (builder.body_count, builder.joint_count, builder.articulation_count)
    expected_counts = (model.body_count, model.joint_count, model.articulation_count)
    if counts != expected_counts:
        raise ValueError(f"{robot_label} topology expected {expected_counts}; found {counts}")

    base_body_index = _unique_short_label(builder.body_label, model.base_body, "body", robot_label)
    foot_body_indices = _named_indices(builder.body_label, model.foot_bodies, "body", robot_label)

    motor_joint_indices: list[int] = []
    motor_coord_indices: list[int] = []
    motor_dof_indices: list[int] = []
    for name in model.motor_joints:
        joint = _unique_short_label(builder.joint_label, name, "joint", robot_label)
        if builder.joint_type[joint] != newton.JointType.REVOLUTE:
            raise ValueError(f"{robot_label} motor joint {name!r} must be revolute")
        if _joint_width(builder.joint_q_start, joint, builder.joint_coord_count) != 1:
            raise ValueError(f"{robot_label} motor joint {name!r} must expose exactly one coordinate")
        if _joint_width(builder.joint_qd_start, joint, builder.joint_dof_count) != 1:
            raise ValueError(f"{robot_label} motor joint {name!r} must expose exactly one DOF")

        coord = builder.joint_q_start[joint]
        dof = builder.joint_qd_start[joint]
        effort_limit = builder.joint_effort_limit[dof]
        if effort_limit != profile.actuator.effort_limit:
            raise ValueError(
                f"{robot_label} motor joint {name!r} has effort limit {effort_limit}; "
                f"expected {profile.actuator.effort_limit}"
            )
        motor_joint_indices.append(joint)
        motor_coord_indices.append(coord)
        motor_dof_indices.append(dof)

    loop_joint_indices = _named_indices(builder.joint_label, model.loop_joints, "joint", robot_label)
    articulated_loops = [index for index in loop_joint_indices if builder.joint_articulation[index] != -1]
    if articulated_loops:
        raise ValueError(
            f"{robot_label} loop-closing joints must be outside the articulation; "
            f"found articulated joint indices {articulated_loops}"
        )

    return RobotLayout(
        motor_joint_names=model.motor_joints,
        base_body_index=base_body_index,
        motor_joint_indices=tuple(motor_joint_indices),
        motor_coord_indices=tuple(motor_coord_indices),
        motor_dof_indices=tuple(motor_dof_indices),
        foot_body_indices=foot_body_indices,
        loop_joint_indices=loop_joint_indices,
    )


def _build_robot_builder(
    profile: RobotProfile,
    *,
    load_visual_shapes: bool = True,
) -> tuple[newton.ModelBuilder, RobotLayout]:
    """Load, validate, and configure one robot asset."""
    asset_path = profile.usd_path
    robot_label = profile.label
    actuator = profile.actuator
    if not asset_path.is_file():
        raise FileNotFoundError(f"{robot_label} asset does not exist: {asset_path}")
    verify_robot_asset(profile)
    for name, value in (
        ("motor_velocity_limit", actuator.simulation_velocity_limit),
        ("motor_armature", actuator.armature),
        ("motor_viscous_friction", actuator.passive_damping),
    ):
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative")

    newton.use_coord_layout_targets = True
    builder = newton.ModelBuilder()
    newton.solvers.SolverKamino.register_custom_attributes(builder)
    builder.request_contact_attributes("force")
    builder.add_usd(
        str(asset_path),
        collapse_fixed_joints=False,
        enable_self_collisions=profile.model.enable_self_collisions,
        hide_collision_shapes=True,
        load_visual_shapes=False,
    )
    if load_visual_shapes:
        visual_builder = newton.ModelBuilder()
        visual_builder.add_usd(
            str(asset_path),
            collapse_fixed_joints=False,
            enable_self_collisions=profile.model.enable_self_collisions,
            hide_collision_shapes=True,
            load_visual_shapes=True,
        )
        _add_merged_visual_shapes(builder, visual_builder)

    layout = _classify_layout(builder, profile)
    builder.joint_target_mode[:] = [int(newton.JointTargetMode.NONE)] * builder.joint_dof_count
    for dof in layout.motor_dof_indices:
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.EFFORT)
        builder.joint_armature[dof] = actuator.armature
        builder.joint_damping[dof] = actuator.passive_damping
        builder.joint_velocity_limit[dof] = actuator.simulation_velocity_limit
    builder.shape_material_mu[:] = [profile.model.dynamic_friction] * builder.shape_count
    builder.shape_material_restitution[:] = [profile.model.restitution] * builder.shape_count
    return builder, layout


def _expand_indices(
    source_indices: tuple[int, ...],
    source_count: int,
    num_worlds: int,
    device: str | torch.device,
) -> torch.Tensor:
    values = [world * source_count + index for world in range(num_worlds) for index in source_indices]
    return torch.tensor(values, dtype=torch.long, device=device).reshape(num_worlds, len(source_indices))


def _validate_world_ownership(model: newton.Model, layout: BatchedLayout) -> None:
    body_world = torch.as_tensor(model.body_world.numpy(), dtype=torch.long, device=layout.base_body_indices.device)
    joint_world = torch.as_tensor(model.joint_world.numpy(), dtype=torch.long, device=layout.base_body_indices.device)
    expected_worlds = torch.arange(layout.num_worlds, device=layout.base_body_indices.device).unsqueeze(1)

    for indices in (layout.base_body_indices.unsqueeze(1), layout.foot_body_indices):
        if not torch.equal(body_world[indices], expected_worlds.expand_as(indices)):
            raise ValueError("Finalized robot body semantic indices do not agree with model.body_world")
    for indices in (layout.motor_joint_indices, layout.loop_joint_indices):
        if not torch.equal(joint_world[indices], expected_worlds.expand_as(indices)):
            raise ValueError("Finalized robot joint semantic indices do not agree with model.joint_world")


def build_batched_model(
    profile: RobotProfile,
    num_worlds: int,
    device: str | torch.device,
    *,
    load_visual_shapes: bool = True,
) -> tuple[newton.Model, BatchedLayout]:
    """Replicate a validated robot and finalize a public Newton model."""
    if num_worlds <= 0:
        raise ValueError(f"num_worlds must be positive; got {num_worlds}")

    robot_builder, source = _build_robot_builder(
        profile,
        load_visual_shapes=load_visual_shapes,
    )
    scene = newton.ModelBuilder()
    newton.solvers.SolverKamino.register_custom_attributes(scene)
    scene.request_contact_attributes("force")
    for _ in range(num_worlds):
        scene.add_world(robot_builder)
    scene.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(
            mu=profile.model.ground_friction,
            restitution=profile.model.restitution,
        )
    )

    model = scene.finalize(device=device, skip_validation_joints=True)
    model.set_gravity((0.0, 0.0, profile.config.physics.gravity))
    body_count = robot_builder.body_count
    joint_count = robot_builder.joint_count
    layout = BatchedLayout(
        num_worlds=num_worlds,
        motor_joint_names=source.motor_joint_names,
        base_body_indices=_expand_indices((source.base_body_index,), body_count, num_worlds, device).squeeze(1),
        motor_joint_indices=_expand_indices(source.motor_joint_indices, joint_count, num_worlds, device),
        motor_coord_indices=_expand_indices(
            source.motor_coord_indices, robot_builder.joint_coord_count, num_worlds, device
        ),
        motor_dof_indices=_expand_indices(source.motor_dof_indices, robot_builder.joint_dof_count, num_worlds, device),
        foot_body_indices=_expand_indices(source.foot_body_indices, body_count, num_worlds, device),
        loop_joint_indices=_expand_indices(source.loop_joint_indices, joint_count, num_worlds, device),
    )
    _validate_world_ownership(model, layout)
    return model, layout
