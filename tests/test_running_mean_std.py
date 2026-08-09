import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


torch = pytest.importorskip("torch")

from BrainCo_DexHand.algo.hora.models.running_mean_std import RunningMeanStd


def test_running_mean_std_updates_on_cpu():
    normalizer = RunningMeanStd((3,))
    inputs = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])

    output = normalizer(inputs)

    assert output.device == inputs.device
    assert normalizer.running_mean.device == inputs.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_running_mean_std_follows_cuda_input_device():
    normalizer = RunningMeanStd((3,))
    inputs = torch.tensor(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        device="cuda:0",
    )

    output = normalizer(inputs)

    assert output.device == inputs.device
    assert normalizer.running_mean.device == inputs.device
    assert normalizer.running_var.device == inputs.device
    assert normalizer.count.device == inputs.device
