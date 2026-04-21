import importlib.util
import sys
import types
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"Using `TRANSFORMERS_CACHE` is deprecated.*",
    category=FutureWarning,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENTS_DIR = REPO_ROOT / "topreward" / "clients"

clients_pkg = types.ModuleType("topreward.clients")
clients_pkg.__path__ = [str(CLIENTS_DIR)]
sys.modules.setdefault("topreward.clients", clients_pkg)

base_spec = importlib.util.spec_from_file_location("topreward.clients.base", CLIENTS_DIR / "base.py")
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["topreward.clients.base"] = base_module
assert base_spec.loader is not None
base_spec.loader.exec_module(base_module)

qwen_spec = importlib.util.spec_from_file_location("test_topreward_qwen_module", CLIENTS_DIR / "qwen.py")
qwen_module = importlib.util.module_from_spec(qwen_spec)
assert qwen_spec.loader is not None
qwen_spec.loader.exec_module(qwen_module)
QwenClient = qwen_module.QwenClient


def test_aligned_video_indices_are_prefix_stable():
    full = QwenClient._aligned_video_indices(total_frames=100, raw_fps=10.0)
    shorter = [idx for idx in full if idx < 40]
    longer = [idx for idx in full if idx < 80]

    assert shorter
    assert longer[: len(shorter)] == shorter


def test_aligned_video_indices_use_expected_cadence():
    indices = QwenClient._aligned_video_indices(total_frames=20, raw_fps=10.0)

    assert indices == [0, 5, 10, 15]


def test_aligned_video_indices_keep_all_frames_when_raw_fps_is_low():
    indices = QwenClient._aligned_video_indices(total_frames=6, raw_fps=1.0)

    assert indices == list(range(6))
