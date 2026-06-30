from mars_core.mars_runner import hash_rows, split_dataset, split_info


def rows(n=10):
    return [{"sample_id": i, "question": f"q{i}", "answer": "A"} for i in range(n)]


def test_paper_mode_reuses_same_rows():
    splits = split_dataset(rows(), "paper_mode", 42)
    info = split_info(splits, "paper_mode", 42)
    assert info["splits"]["opt"]["hash"] == info["splits"]["test"]["hash"]


def test_strict_mode_splits_rows():
    splits = split_dataset(rows(), "strict_mode", 42)
    assert len(splits["opt"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0
    assert hash_rows(splits["opt"]) != hash_rows(splits["test"])
