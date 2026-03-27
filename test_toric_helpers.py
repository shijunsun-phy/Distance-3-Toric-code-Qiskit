from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toric_qec.core import binary_rank, build_logicals, build_toric_pcm


def test_distance_3_dimensions():
    code, Hx, Hz = build_toric_pcm(3)
    LX, LZ = build_logicals(code)

    assert code.n == 18
    assert Hx.shape == (9, 18)
    assert Hz.shape == (9, 18)
    assert LX.shape == (2, 18)
    assert LZ.shape == (2, 18)
    assert code.n - binary_rank(Hx) - binary_rank(Hz) == 2
