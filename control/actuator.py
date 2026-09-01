"""Device-resident Robstride actuator network."""

from __future__ import annotations

import numpy as np
import warp as wp
from warp_nn import nn

from control.config import RobotProfile, verify_actuator_asset


class Actuator:
    """Load and execute the shared actuator network with warp-nn."""

    history_length = 6

    def __init__(self, profile: RobotProfile, device: str | wp.Device) -> None:
        self.path = verify_actuator_asset(profile)
        self.sha256 = profile.actuator_sha256
        self.device = wp.get_device(device)
        self.num_joints = profile.action_dim
        self.network = nn.Sequential(
            nn.Linear(2 * self.history_length, 64),
            nn.SoftSign(),
            nn.Linear(64, 64),
            nn.SoftSign(),
            nn.Linear(64, 1),
        ).to(self.device)
        with np.load(self.path) as weights:
            state = {
                name: np.asarray(value).reshape((-1, 1)) if name.endswith(".bias") else np.asarray(value)
                for name, value in weights.items()
            }
        self.network.load_state_dict(state)

    def __call__(self, network_input: wp.array) -> wp.array:
        if network_input.shape != (self.num_joints, 2 * self.history_length):
            raise ValueError(
                f"actuator input must have shape ({self.num_joints}, {2 * self.history_length}); "
                f"got {network_input.shape}"
            )
        return self.network(network_input)
