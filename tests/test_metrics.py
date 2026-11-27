from rag import metrics as M


def test_normalize():
    assert M.normalize_answer("The Answer, is: A Test!") == "answer is test"


def test_exact_match_and_f1():
    assert M.exact_match("The cosine similarity.", "cosine similarity") == 1.0
    assert M.f1_score("cosine similarity score", "cosine similarity") == 0.8
    assert M.f1_score("banana", "cosine similarity") == 0.0


def test_contains_and_evidence():
    assert M.contains_gold("It is 384 dimensions", "384") == 1.0
    assert M.evidence_in_text("Patrick  Lewis", "by PATRICK LEWIS and colleagues")
    assert not M.evidence_in_text("xyz", "abc")


def test_reciprocal_rank():
    assert M.reciprocal_rank(None) == 0.0
    assert M.reciprocal_rank(4) == 0.25
