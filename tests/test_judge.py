import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from judge import expand, extract_citations  # noqa: E402


def test_expand_handles_lists_and_ranges():
    assert expand("E3") == ["E3"]
    assert expand("E3, E5") == ["E3", "E5"]
    assert expand("E3–E6") == ["E3", "E4", "E5", "E6"]
    assert expand("E10-12") == ["E10", "E11", "E12"]


def test_extract_citations_pairs_sentences_with_ids():
    answer = "Solar rose to 8.6% in 2025 [E1]. Wind was 10.3% [E2][E3]. No citation here. Gas led [E4, E5]."
    assert extract_citations(answer) == [
        ("Solar rose to 8.6% in 2025 [E1].", ["E1"]),
        ("Wind was 10.3% [E2][E3].", ["E2", "E3"]),
        ("Gas led [E4, E5].", ["E4", "E5"]),
    ]
