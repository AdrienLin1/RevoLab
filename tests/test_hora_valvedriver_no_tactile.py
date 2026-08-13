"""Static integration tests for non-TacSL HORA ValveDriver tasks."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw"
)
REGISTRY_PATH = TASK_DIR / "__init__.py"
TRAIN_PATH = REPO_ROOT / "scripts/hora/train.py"
PLAY_PATH = REPO_ROOT / "scripts/hora/play.py"


def test_non_tactile_valvedriver_gym_tasks_use_the_base_screw_environment():
    """Register both valve sizes without routing through the TacSL environment."""
    registry = REGISTRY_PATH.read_text(encoding="utf-8")

    for task_id in (
        "BrainCo-Direct-Revo3-HoraValveDriver-v0",
        "BrainCo-Direct-Revo3-HoraValveDriver25-v0",
        "BrainCo-Direct-Revo3-HoraValveDriver40-v0",
        "RevoHoraValveDriver-v0",
        "RevoHoraValveDriver25-v0",
        "RevoHoraValveDriver40-v0",
    ):
        block = registry[registry.index(f'id="{task_id}"') :]
        block = block[: block.index("\n)\n")]
        assert "revo3_hand_screw_env:Revo3HandScrewEnv" in block
        assert "revo3_hand_screw_tactile_env" not in block

    assert "revo3_hand_screw_env_cfg:Revo3HandVavleDriverEnvCfg" in registry
    assert "revo3_hand_screw_env_cfg:Revo3HandValveDriver25EnvCfg" in registry
    assert "revo3_hand_screw_env_cfg:Revo3HandValveDriver40EnvCfg" in registry


def test_train_routes_valvedriver_through_hora_ppo_and_proprio_adapt():
    """Expose ValveDriver through the standard non-tactile HORA pipeline."""
    train = TRAIN_PATH.read_text(encoding="utf-8")

    assert "'valvedriver', 'valvedriver_25', 'valvedriver_40'" in train
    assert "'valvedriver': Revo3HandVavleDriverEnvCfg" in train
    assert "'valvedriver_25': Revo3HandValveDriver25EnvCfg" in train
    assert "'valvedriver_40': Revo3HandValveDriver40EnvCfg" in train
    assert "args.train_cfg = 'Revo3HandScrew' if args.task in _SCREW_TASKS" in train
    assert "env_cfg.enable_contact_in_obs = False" in train
    assert "agent_cls = TactileDAgger" in train


def test_play_supports_non_tactile_stage1_and_stage2_valve_checkpoints():
    """Keep evaluation available for both teacher and adapted student policies."""
    play = PLAY_PATH.read_text(encoding="utf-8")

    assert "'valvedriver': Revo3HandVavleDriverEnvCfg" in play
    assert "'valvedriver_25': Revo3HandValveDriver25EnvCfg" in play
    assert "'valvedriver_40': Revo3HandValveDriver40EnvCfg" in play
    assert "from BrainCo_DexHand.algo.hora.padapt.padapt import ProprioAdapt" in play
    assert "policy_mode = 'proprio_student'" in play
    assert "'proprio_hist': agent.sa_mean_std" in play
    assert "env_cfg.enable_contact_in_obs = False" in play
