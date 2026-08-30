import pytest
import torch

from control.config import default_asset_root, discover_robot_profiles
from sim.simulation import KaminoSimulation

PROFILES, _ = discover_robot_profiles(default_asset_root())


def test_cpu_runtime_resets_and_steps_every_profile() -> None:
    for profile in PROFILES.values():
        simulation = KaminoSimulation(profile, 1, "cpu", load_visual_shapes=False)
        mask = torch.ones(1, dtype=torch.bool)

        assert torch.equal(simulation.reset(mask), mask)
        view = simulation.step_motor_torques(torch.zeros(1, profile.action_dim))
        assert torch.isfinite(view.base_pose).all()
        assert torch.isfinite(view.loop_position_error).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_graph_steps_every_profile() -> None:
    for profile in PROFILES.values():
        simulation = KaminoSimulation(profile, 1, "cuda:0", load_visual_shapes=False, use_cuda_graph=True)

        assert simulation.cuda_graph_enabled
        view = simulation.step_motor_torques(torch.zeros(1, profile.action_dim, device="cuda:0"))
        assert view.base_pose.device.type == "cuda"
        assert torch.isfinite(view.base_pose).all()
