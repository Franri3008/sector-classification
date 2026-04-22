from src.utils.hashing import normalize_tag, stable_hash, tag_id


def test_stable_hash_deterministic():
    assert stable_hash("hello") == stable_hash("hello")
    assert stable_hash("hello") != stable_hash("HELLO")


def test_normalize_tag_collapses_whitespace_and_case():
    assert normalize_tag("  Hello   World  ") == "hello world"
    assert normalize_tag("FoO") == "foo"


def test_tag_id_is_normalized():
    assert tag_id("AI") == tag_id("ai ")
    assert tag_id("AI") == tag_id("  ai  ")
    assert tag_id("AI") != tag_id("ml")
