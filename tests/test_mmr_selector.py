from src.ai.diversity.mmr_selector import MMRSelector


def _doc(name: str, embedding):
    return {"content": name, "metadata": {"source": name}, "embedding": embedding}


def test_empty_input_returns_empty():
    assert MMRSelector().select([], top_k=5, lambda_mult=0.6) == []


def test_pure_relevance_keeps_input_order():
    docs = [
        _doc("a", [1.0, 0.0]),
        _doc("b", [1.0, 0.0]),  # duplicate of a
        _doc("c", [0.0, 1.0]),
    ]
    selected = MMRSelector().select(docs, top_k=2, lambda_mult=1.0)
    assert [d["content"] for d in selected] == ["a", "b"]


def test_diversity_skips_redundant_duplicate():
    docs = [
        _doc("a", [1.0, 0.0]),
        _doc("b", [1.0, 0.0]),  # identical to a -> redundant
        _doc("c", [0.0, 1.0]),  # orthogonal -> diverse
    ]
    selected = MMRSelector().select(docs, top_k=2, lambda_mult=0.5)
    contents = [d["content"] for d in selected]
    assert contents[0] == "a"
    assert contents[1] == "c"


def test_pure_diversity_prefers_orthogonal_over_near_duplicate():
    docs = [
        _doc("a", [1.0, 0.0]),
        _doc("b", [0.98, 0.2]),  # near duplicate of a
        _doc("c", [0.0, 1.0]),   # orthogonal
    ]
    selected = MMRSelector().select(docs, top_k=2, lambda_mult=0.0)
    assert selected[1]["content"] == "c"


def test_top_k_larger_than_input_returns_all():
    docs = [_doc("a", [1.0, 0.0]), _doc("b", [0.0, 1.0])]
    selected = MMRSelector().select(docs, top_k=10, lambda_mult=0.6)
    assert len(selected) == 2


def test_zero_vector_does_not_raise():
    docs = [_doc("a", [0.0, 0.0]), _doc("b", [1.0, 0.0])]
    selected = MMRSelector().select(docs, top_k=2, lambda_mult=0.6)
    assert len(selected) == 2
