"""Static integration tests for the nominal 25 mm tactile valve task."""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = (
    REPO_ROOT
    / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/direct/hora_screw"
)
URDF_PATH = REPO_ROOT / "assets/urdf/screw/vavledriver/valvedriver_hex_25.urdf"
ASSETS_PATH = TASK_DIR / "assets.py"
ENV_CFG_PATH = TASK_DIR / "revo3_hand_screw_env_cfg.py"
TACTILE_CFG_PATH = TASK_DIR / "revo3_hand_screw_tactile_env_cfg.py"
REGISTRY_PATH = TASK_DIR / "__init__.py"
TRAIN_PATH = REPO_ROOT / "scripts/hora/train.py"
PLAY_PATH = REPO_ROOT / "scripts/hora/play.py"


def _class_literal(path: Path, class_name: str, attribute: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assignment = next(
        node
        for node in class_node.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == attribute
    )
    return ast.literal_eval(assignment.value)


def test_valvedriver_25_urdf_has_25_mm_nominal_circumradius():
    """Scale the 30 mm source mesh by 5/6 in both visual and collision."""
    root = ET.parse(URDF_PATH).getroot()
    valve = next(link for link in root.findall("link") if link.attrib["name"] == "valve")
    visual_mesh = valve.find("visual/geometry/mesh")
    collision_mesh = valve.find("collision/geometry/mesh")
    expected_scale = 25.0 / 30.0

    for mesh in (visual_mesh, collision_mesh):
        assert mesh is not None
        scale = tuple(float(value) for value in mesh.attrib["scale"].split())
        assert scale == pytest.approx((expected_scale, expected_scale, 1.0))


def test_valvedriver_tactile25_uses_25_mm_asset_and_reward_radius():
    """Connect the 25 mm URDF to matching base and tactile config classes."""
    assets = ASSETS_PATH.read_text(encoding="utf-8")
    base_cfg = ENV_CFG_PATH.read_text(encoding="utf-8")

    assert '"valvedriver_hex_25.urdf"' in assets
    assert "SCREW_VALVE_DRIVER_25_CFG" in assets
    assert "class Revo3HandValveDriver25EnvCfg" in base_cfg
    assert "self.object_cfg = SCREW_VALVE_DRIVER_25_CFG" in base_cfg
    assert _class_literal(
        TACTILE_CFG_PATH,
        "Revo3HandValveDriver25TactileEnvCfg",
        "coord_obj_radius",
    ) == pytest.approx(0.025)


def test_train_and_play_accept_exact_valvedriver_tactile25_name():
    """Route the requested CLI spelling through the tactile environment."""
    for path in (TRAIN_PATH, PLAY_PATH):
        source = path.read_text(encoding="utf-8")
        assert "'valvedriver_tactile25'" in source
        assert (
            "'valvedriver_tactile25': Revo3HandValveDriver25TactileEnvCfg"
            in source
        )


def test_valvedriver_tactile25_has_long_and_short_gym_ids():
    """Expose the new task through both Gym naming conventions."""
    registry = REGISTRY_PATH.read_text(encoding="utf-8")
    for task_id in (
        "BrainCo-Direct-Revo3-HoraValveDriverTactile25-v0",
        "RevoHoraValveDriverTactile25-v0",
    ):
        assert f'id="{task_id}"' in registry
    assert "Revo3HandValveDriver25TactileEnvCfg" in registry
