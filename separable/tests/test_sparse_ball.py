import json

from tpinn import sparse_ball


def test_single_level_matches_full_tensor(tmp_path):
    sparse_ball.main(["--base", "2", "--top", "2", "--workers", "2", "--tuples", "500", "--out", str(tmp_path)])
    rows = json.loads((tmp_path / "classical_ball" / "sparse_a1.0.json").read_text())["rows"]
    assert len(rows) == 1
    assert 0.03 < rows[0]["err_full"] < 0.06
