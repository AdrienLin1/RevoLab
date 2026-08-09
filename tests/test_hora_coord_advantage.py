"""CPU tests for separate task and coordination PPO advantages."""

from pathlib import Path
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_PATH = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(EXTENSION_PATH) not in sys.path:
    sys.path.insert(0, str(EXTENSION_PATH))

from BrainCo_DexHand.algo.hora.ppo.experience import ExperienceBuffer
from BrainCo_DexHand.algo.hora.ppo.ppo import PPO


def _coord_buffer() -> ExperienceBuffer:
    """Build a compact buffer with separate coordination storage."""
    return ExperienceBuffer(
        num_envs=2,
        horizon_length=2,
        batch_size=4,
        minibatch_size=2,
        obs_dim=1,
        act_dim=1,
        priv_dim=1,
        device=torch.device("cpu"),
        separate_coord_advantage=True,
    )


def test_separate_gae_normalizes_and_combines_advantages():
    """Combine independently normalized advantages with the configured beta."""
    buffer = _coord_buffer()
    buffer.storage_dict["rewards"].copy_(
        torch.tensor([[[1.0], [3.0]], [[2.0], [6.0]]])
    )
    buffer.storage_dict["coord_rewards"].copy_(
        torch.tensor([[[0.0], [2.0]], [[1.0], [4.0]]])
    )
    buffer.storage_dict["values"].zero_()
    buffer.storage_dict["coord_values"].zero_()
    buffer.storage_dict["dones"].zero_()

    last_values = torch.zeros(2, 1)
    buffer.computer_return(
        last_values,
        gamma=0.0,
        tau=0.0,
        coord_last_values=last_values,
    )
    data = buffer.prepare_training(
        normalize_advantage=True,
        coord_advantage_coef=0.25,
    )

    assert data["task_advantages"].mean().item() == pytest.approx(0.0, abs=1.0e-6)
    assert data["task_advantages"].std(unbiased=False).item() == pytest.approx(1.0)
    assert data["coord_advantages"].mean().item() == pytest.approx(0.0, abs=1.0e-6)
    assert data["coord_advantages"].std(unbiased=False).item() == pytest.approx(1.0)
    assert torch.allclose(
        data["advantages"],
        data["task_advantages"] + 0.25 * data["coord_advantages"],
    )
    assert buffer.rollout_stats["coord_reward_nonzero_ratio"] == pytest.approx(0.75)
    assert isinstance(buffer[0], dict)


def test_constant_coord_advantage_stays_zero():
    """Avoid turning a constant coordination return into numerical noise."""
    buffer = _coord_buffer()
    buffer.storage_dict["rewards"].copy_(
        torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    )
    buffer.storage_dict["coord_rewards"].fill_(0.5)
    buffer.storage_dict["values"].zero_()
    buffer.storage_dict["coord_values"].zero_()
    buffer.storage_dict["dones"].zero_()

    last_values = torch.zeros(2, 1)
    buffer.computer_return(
        last_values,
        gamma=0.0,
        tau=0.0,
        coord_last_values=last_values,
    )
    data = buffer.prepare_training(coord_advantage_coef=0.2)

    assert torch.count_nonzero(data["coord_advantages"]) == 0
    assert torch.allclose(data["advantages"], data["task_advantages"])


def test_coord_actor_coefficient_uses_a_fixed_step_schedule():
    """Anneal beta from its early guide weight to its final task-safe weight."""
    trainer = PPO.__new__(PPO)
    trainer.coord_advantage_curriculum_start = 0
    trainer.coord_advantage_curriculum_end = 100
    trainer.coord_advantage_coef_initial = 0.20
    trainer.coord_advantage_coef_final = 0.05

    trainer.agent_steps = 0
    assert trainer._coord_advantage_coef() == pytest.approx(0.20)
    trainer.agent_steps = 50
    assert trainer._coord_advantage_coef() == pytest.approx(0.125)
    trainer.agent_steps = 100
    assert trainer._coord_advantage_coef() == pytest.approx(0.05)


def test_policy_gradient_stats_measure_scaled_conflict():
    """Report actual beta-scaled gradient size and task/coordination alignment."""
    trainer = PPO.__new__(PPO)
    trainer.device = torch.device("cpu")
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    prediction = trainer.model(torch.ones(1, 1)).mean()

    (prediction - 0.2 * prediction).backward(retain_graph=True)
    trainer._record_policy_gradient_stats(
        task_actor_loss=prediction,
        coord_actor_loss=-prediction,
        coord_coef=0.2,
    )

    assert trainer.coord_gradient_stats["task_policy_grad_norm"] == pytest.approx(1.0)
    assert trainer.coord_gradient_stats["coord_policy_grad_norm"] == pytest.approx(0.2)
    assert trainer.coord_gradient_stats["task_coord_grad_cosine"] == pytest.approx(-1.0)


def test_per_env_column_rejects_aggregated_reward_info():
    """Reject scalar extras instead of allowing accidental broadcast across envs."""
    trainer = PPO.__new__(PPO)
    trainer.device = torch.device("cpu")

    column = trainer._as_per_env_column(
        torch.arange(4.0), "coord_reward", 4, torch.float32
    )
    assert column.shape == (4, 1)

    with pytest.raises(RuntimeError, match="one value per environment"):
        trainer._as_per_env_column(torch.tensor(1.0), "coord_reward", 4, torch.float32)



def test_rollout_coord_value_does_not_retain_an_autograd_graph():
    """Keep external coordination critic predictions detached during rollout."""
    trainer = PPO.__new__(PPO)
    trainer.device = torch.device("cpu")
    trainer.running_mean_std = torch.nn.Identity()
    trainer.normalize_value = False
    trainer.separate_coord_advantage = True
    trainer.coord_value_head = torch.nn.Linear(3, 1)

    class RolloutModel:
        def act(self, input_dict):
            batch_size = input_dict["obs"].shape[0]
            return {
                "values": torch.zeros(batch_size, 1),
                "features": torch.ones(batch_size, 3),
            }

    trainer.model = RolloutModel()
    result = trainer.model_act(
        {
            "obs": torch.ones(4, 2),
            "priv_info": torch.ones(4, 1),
        }
    )

    assert result["coord_values"].shape == (4, 1)
    assert not result["coord_values"].requires_grad
    assert result["coord_values"].grad_fn is None


def test_single_return_resume_drops_only_legacy_coord_optimizer_state():
    """Preserve actor Adam moments when removing the old coordination critic."""
    model = torch.nn.Linear(2, 1)
    coord_head = torch.nn.Linear(1, 1)
    legacy_optimizer = torch.optim.Adam(
        list(model.parameters()) + list(coord_head.parameters()), lr=1.0e-3
    )
    loss = model(torch.ones(1, 2)).square().mean()
    loss = loss + coord_head(torch.ones(1, 1)).square().mean()
    loss.backward()
    legacy_optimizer.step()
    saved_state = legacy_optimizer.state_dict()

    trainer = PPO.__new__(PPO)
    trainer.separate_coord_advantage = False
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    trainer._load_optimizer_state(saved_state, has_coord_state=True)

    loaded_state = trainer.optimizer.state_dict()
    assert len(loaded_state["param_groups"][0]["params"]) == 2
    assert len(loaded_state["state"]) == 2
    assert all("exp_avg" in state for state in loaded_state["state"].values())
